import networkx as nx
import numpy as np
import torch
import random
import torch.utils.data

from functools import cmp_to_key

import util

class GraphSampler(torch.utils.data.Dataset):
    ''' Sample graphs and nodes in graph
    '''
    def __init__(self, G_list, features='default', normalize=True, max_num_nodes=0, sort_type='degree1'):
        self.adj_all = []
        self.graph_seq_all = []
        self.len_all = []
        self.feature_all = []  # 按seq中id选择feature的列表
        self.seq_feature_all = []
        self.label_all = []

        if max_num_nodes == 0:
            # find the max number of nodes in all graph list.
            self.max_num_nodes = max([G.number_of_nodes() for G in G_list])  # 选择所有图中最多的节点数量，在一张图中选最大的就是节点数量，已经在load的时候重新映射过了。
            print("The max node id : ", self.max_num_nodes)
        else:
            self.max_num_nodes = max_num_nodes
            print("Predefined max node id : ",self.max_num_nodes)

        # get the dimension of feature. just look at the first one:-)
        self.feat_dim = util.node_dict(G_list[0])[0]['feat'].shape[0]

        for G in G_list:
            adj = np.array(nx.to_numpy_matrix(G))
            if normalize:
                sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
                adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
            self.adj_all.append(adj)

            # get the graph sequence
            seq = self.graph2seq(G, sort_type)
            self.graph_seq_all.append(seq)

            self.len_all.append(G.number_of_nodes())
            self.label_all.append(G.graph['label'])

            # feat matrix: max_num_nodes x feat_dim
            if features == 'default':  # 图中节点的feature
                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i,u in enumerate(G.nodes()):
                    f[i,:] = util.node_dict(G)[u]['feat']
                self.feature_all.append(f)
            elif features == 'id':
                self.feature_all.append(np.identity(self.max_num_nodes))
            elif features == 'deg-num':
                degs = np.sum(np.array(adj), 1)
                degs = np.expand_dims(np.pad(degs, [0, self.max_num_nodes - G.number_of_nodes()], 0),
                                      axis=1)
                self.feature_all.append(degs)
            elif features == 'deg':
                self.max_deg = 10
                degs = np.sum(np.array(adj), 1).astype(int)
                degs[degs>self.max_deg] = self.max_deg
                feat = np.zeros((len(degs), self.max_deg + 1))
                feat[np.arange(len(degs)), degs] = 1
                feat = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                        'constant', constant_values=0)

                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i,u in enumerate(util.node_iter(G)):
                    f[i,:] = util.node_dict(G)[u]['feat']

                feat = np.concatenate((feat, f), axis=1)

                self.feature_all.append(feat)
            elif features == 'struct':
                self.max_deg = 10
                degs = np.sum(np.array(adj), 1).astype(int)
                degs[degs>10] = 10
                feat = np.zeros((len(degs), self.max_deg + 1))
                feat[np.arange(len(degs)), degs] = 1
                degs = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                        'constant', constant_values=0)

                clusterings = np.array(list(nx.clustering(G).values()))
                clusterings = np.expand_dims(np.pad(clusterings, 
                                                    [0, self.max_num_nodes - G.number_of_nodes()],
                                                    'constant'),
                                             axis=1)
                g_feat = np.hstack([degs, clusterings])
                if 'feat' in util.node_dict(G)[0]:
                    node_feats = np.array([util.node_dict(G)[i]['feat'] for i in range(G.number_of_nodes())])
                    node_feats = np.pad(node_feats, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                                        'constant')
                    g_feat = np.hstack([g_feat, node_feats])

                self.feature_all.append(g_feat)

            seq_feats = []  # adding sequence features
            for u in seq:
                seq_feats.append(self.feature_all[-1][u])

            self.seq_feature_all.append(seq_feats)

        self.feat_dim = self.feature_all[0].shape[1]

    @staticmethod
    def graph2seq(G, sort_type):
        def cmp(n1, n2):
            if G.degree(n1) > G.degree(n2):
                return 1
            if G.degree(n1) < G.degree(n2):
                return -1
            if G.degree(n1) == G.degree(n2):
                return 0

        if sort_type == 'degree1':  # max->min
            return sorted(G.nodes(), key=cmp_to_key(cmp), reverse=1)
        elif sort_type == 'degree0':  # min->max
            return sorted(G.nodes(), key=cmp_to_key(cmp), reverse=0)
        elif sort_type == 'bfs':
            root_node = sorted(G.nodes(), key=cmp_to_key(cmp))[-1]
            return list(nx.bfs_tree(G, root_node))
        elif sort_type == 'bfs_r':
            root_node = random.choice(list(G.nodes()))
            return list(nx.bfs_tree(G, root_node))
        elif sort_type == 'dfs':
            root_node = sorted(G.nodes(), key=cmp_to_key(cmp))[-1]
            return list(nx.dfs_tree(G, root_node))
        elif sort_type == 'dfs_r':
            root_node = random.choice(list(G.nodes()))
            return list(nx.dfs_tree(G, root_node))
        else:
            print('error sort type. using degree1 as default.')
            return sorted(G.nodes(), key=cmp_to_key(cmp), reverse=1)


    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        graph_seq = self.graph_seq_all[idx]
        num_nodes = len(graph_seq)
        graph_seq_padded = np.zeros(self.max_num_nodes)
        graph_seq_padded[:num_nodes] = graph_seq  # padding图的序列

        seq_feats = self.seq_feature_all[idx]
        num_nodes = len(seq_feats)
        seq_feats_padded = np.zeros((self.max_num_nodes, self.feat_dim))
        seq_feats_padded[:num_nodes,:] = seq_feats  # padding图的序列

        # use all nodes for aggregation (baseline)

        return {'sequence': graph_seq_padded,  # node id of sequence
                'seq_feats': seq_feats_padded,  # self.seq_feature_all[idx].copy(),
                # 'feats': self.feature_all[idx].copy(),  # 图中节点的属性矩阵 max_num_nodes x feat_dim
                'label': self.label_all[idx],  # 图的label，ground-truth
                'num_nodes': num_nodes  # 单个图中节点的数量 （真实数量，没有padding过的）
                }

