import networkx as nx
import numpy as np
import random
import torch.utils.data

from functools import cmp_to_key

import util

class GraphSampler(torch.utils.data.Dataset):
    ''' Sample graphs and nodes in graph
    '''
    def __init__(self, G_list, features='default', max_num_nodes=0, sort_type='degree1', cls_flag=False):
        # according to the order in sequences
        self.adj_all = []
        self.graph_seq_all = []  # all graph sequences
        self.len_all = []   # the node number of graphs (sequences)
        self.seq_feature_all = []

        self.label_all = []

        self.sort_type = sort_type
        self.cls_flag = cls_flag

        if max_num_nodes == 0:
            # find the max number of nodes in all graph list.
            self.max_num_nodes = max([G.number_of_nodes() for G in G_list])
            print("The max node id : ", self.max_num_nodes)
        else:
            self.max_num_nodes = max_num_nodes
            print("Predefined max node id : ",self.max_num_nodes)

        # get the dimension of feature. just look at the first one:-)
        self.feat_dim = util.node_dict(G_list[0])[0]['feat'].shape[0]

        for G in G_list:

            # first, get the graph sequence
            seq_size, seq = self.graph2seq(G, self.sort_type)
            self.graph_seq_all.append(seq)

            # generate adj according to sequence
            if seq_size > 0:
                adj = [np.array(nx.to_numpy_matrix(G, nodelist=seq_item)) for seq_item in seq]
            else:
                adj = np.array(nx.to_numpy_matrix(G, nodelist=seq))
            self.adj_all.append(adj)

            self.len_all.append(G.number_of_nodes())
            self.label_all.append(G.graph['label'])

            # feat matrix: max_num_nodes x feat_dim
            if features == 'default':
                if seq_size > 0:
                    f_tem = []
                    for item in seq:
                        f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                        for i,u in enumerate(item):
                            f[i,:] = util.node_dict(G)[u]['feat']
                        f_tem.append(f)
                    f_tem = np.stack(f_tem)
                else:
                    f_tem = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                    for i,u in enumerate(seq):
                        f_tem[i,:] = util.node_dict(G)[u]['feat']
                self.seq_feature_all.append(f_tem)
            else:
                print('Error feature selection!')
                exit(0)

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
            return 0, sorted(G.nodes(), key=cmp_to_key(cmp), reverse=1)
        elif sort_type == 'degree0':  # min->max
            return 0, sorted(G.nodes(), key=cmp_to_key(cmp), reverse=0)
        elif sort_type == 'bfs':
            root_node = sorted(G.nodes(), key=cmp_to_key(cmp))[-1]
            return 0, list(nx.bfs_tree(G, root_node))
        elif sort_type == 'bfs_btw':
            d = nx.betweenness_centrality(G)
            root_node = sorted(d.items(),key=lambda d:d[1])[-1][0]
            return 0, list(nx.bfs_tree(G, root_node))
        elif sort_type == 'bfs_r':
            root_node = random.choice(list(G.nodes()))
            return 0, list(nx.bfs_tree(G, root_node))
        elif sort_type == 'dfs':
            root_node = sorted(G.nodes(), key=cmp_to_key(cmp))[-1]
            return 0, list(nx.dfs_tree(G, root_node))
        elif sort_type == 'dfs_btw':
            d = nx.betweenness_centrality(G)
            root_node = sorted(d.items(), key=lambda d:d[1])[-1][0]
            return 0, list(nx.dfs_tree(G, root_node))
        elif sort_type == 'dfs_r':
            root_node = random.choice(list(G.nodes()))
            return 0, list(nx.dfs_tree(G, root_node))
        elif sort_type == 'all':
            seq_ls = []

            # rank by degree from min to max
            d = nx.degree_centrality(G)
            deg_rank = sorted(d.items(), key=lambda d:d[1])

            # rank by betweenness from min to max
            d = nx.betweenness_centrality(G)
            btw_rank = sorted(d.items(), key=lambda d:d[1])

            # rank by closeness from min to max
            d = nx.closeness_centrality(G)
            cls_rank = sorted(d.items(), key=lambda d:d[1])

            # get the sequences

            # degree max first
            deg_max_node = deg_rank[-1][0]
            seq_ls.append(list(nx.dfs_tree(G, deg_max_node)))
            seq_ls.append(list(nx.bfs_tree(G, deg_max_node)))
            # degree min first
            deg_min_node = deg_rank[0][0]
            seq_ls.append(list(nx.dfs_tree(G, deg_min_node)))
            seq_ls.append(list(nx.bfs_tree(G, deg_min_node)))

            # betweenness max first
            btw_max_node = btw_rank[-1][0]
            seq_ls.append(list(nx.dfs_tree(G, btw_max_node)))
            seq_ls.append(list(nx.bfs_tree(G, btw_max_node)))
            # betweenness min first
            btw_min_node = btw_rank[0][0]
            seq_ls.append(list(nx.dfs_tree(G, btw_min_node)))
            seq_ls.append(list(nx.bfs_tree(G, btw_min_node)))

            # closeness max first
            cls_max_node = cls_rank[-1][0]
            seq_ls.append(list(nx.dfs_tree(G, cls_max_node)))
            seq_ls.append(list(nx.bfs_tree(G, cls_max_node)))
            # closeness min first
            cls_min_node = cls_rank[0][0]
            seq_ls.append(list(nx.dfs_tree(G, cls_min_node)))
            seq_ls.append(list(nx.bfs_tree(G, cls_min_node)))


            return 1, seq_ls
        else:
            print('error sort type. using degree1 as default.')
            return 0, sorted(G.nodes(), key=cmp_to_key(cmp), reverse=1)

    def self_padding(self, adj, graph_seq):
        assert(len(graph_seq) == adj.shape[0])
        num_nodes = len(graph_seq)

        graph_seq_padded = np.zeros(self.max_num_nodes)
        graph_seq_padded[:num_nodes] = graph_seq  # padding图的序列

        #  only adj considers the cls HERE!
        adj_padded = np.zeros((self.max_num_nodes + self.cls_flag, self.max_num_nodes + self.cls_flag))
        if self.cls_flag:
            adj_cls = np.ones([num_nodes+1, num_nodes+1])
            adj_cls[1:, 1:] = adj
            adj = adj_cls

        adj_padded[:num_nodes + self.cls_flag, :num_nodes + self.cls_flag] = adj

        return adj_padded, graph_seq_padded, num_nodes

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        if self.sort_type == 'all':
            num_sequences = len(self.graph_seq_all[idx])
            assert(num_sequences == len(self.adj_all[idx]))

            adj_padded = []
            graph_seq_padded = []

            for index in range(num_sequences):
                graph_seq = self.graph_seq_all[idx][index]
                adj = self.adj_all[idx][index]
                adj_padded_, graph_seq_padded_, num_nodes_ = self.self_padding(adj, graph_seq)

                adj_padded.append(adj_padded_)
                graph_seq_padded.append(graph_seq_padded_)

            adj_padded = np.stack(adj_padded,axis=0)
            graph_seq_padded = np.stack(graph_seq_padded,axis=0)
            num_nodes = num_nodes_

        else:
            graph_seq = self.graph_seq_all[idx]
            adj = self.adj_all[idx]

            adj_padded, graph_seq_padded, num_nodes = self.self_padding(adj, graph_seq)

        # print(adj_padded.shape,graph_seq_padded.shape,num_nodes)

        return {'adj': adj_padded,  # (max_num, max_num)
                'sequence': graph_seq_padded,  # (max_num,)
                'seq_feats': self.seq_feature_all[idx].copy(),  # max_num * feat_dim
                'label': self.label_all[idx],  # 图的label，ground-truth
                'num_nodes': num_nodes  # 单个图中节点的数量 （真实数量，没有padding过的）
                }

