# -*- coding: utf-8 -*-
'''
@Time    : 2021-04-12 9:25 p.m.
@Author  : datasnail
@File    : encoders.py.py
'''

import torch
import math
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:,:-1]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class GSTransformer(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers, nclass, dropout=0.5, fine_tune=False):
        """
        :param feats:  predefined feature matrix
        :param ninp:   the dimension of input
        :param nhead:  the head num
        :param nhid:   the hidden dimension of encoder layer
        :param nlayers:    the number of encoder layer
        :param dropout:
        :param fine_tune:   fine-tune the predefined feature matrix
        """
        super(GSTransformer, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        # self.ninp = feats.size(1)  # 输入维度
        self.seq_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)  # 位置编码层 ninp is the dimension of input
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # emebedding  layer using predefined feats
        # https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222/2
        # self.embedding = nn.Embedding(feats.size(0), feats.size(1))
        # self.embedding.weight = nn.Parameter(feats)
        # self.embedding.weight.requires_grad = fine_tune

        self.pred = self.build_pred_layers(ninp, 0, nclass)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        # nn.init.zeros_(self.decoder.weight)
        # nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        """
        transformer后的预测层
        """
        pred_input_dim = pred_input_dim * num_aggs
        if pred_hidden_dims == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def forward(self, seq_feats, has_mask=False):
        cls = torch.ones(seq_feats.size(0), self.ninp)  #
        # emb = self.embedding(seq)  # ntoken x ninp
        emb = torch.cat([cls, seq_feats], dim=0)

        if has_mask:
            device = emb.device
            if self.seq_mask is None or self.seq_mask.size(0) != len(seq_feats):
                mask = self._generate_square_subsequent_mask(len(seq_feats)).to(device)
                self.seq_mask = mask
        else:
            self.src_mask = None

        emb = self.pos_encoder(emb)
        output = self.transformer_encoder(emb)
        output = self.pred(output[0,:])

        return output

    def loss(self, pred, label):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7

        if type == 'softmax':
            loss = F.cross_entropy(pred, label, reduction='mean')
        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1,1), 1)
            loss = torch.nn.MultiLabelMarginLoss()(pred, label_onehot)

        # if self.linkpred:
        #     max_num_nodes = adj.size()[1]
        #     pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
        #     tmp = pred_adj0
        #     pred_adj = pred_adj0
        #     for adj_pow in range(adj_hop-1):  #这是什么意思？
        #         tmp = tmp @ pred_adj0
        #         pred_adj = pred_adj + tmp
        #     pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).cuda())  # 大于1取1
        #     #print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
        #     #print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
        #     #self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
        #     self.link_loss = -adj * torch.log(pred_adj+eps) - (1-adj) * torch.log(1-pred_adj+eps)
        #     if batch_num_nodes is None:
        #         num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
        #         print('Warning: calculating link pred loss without masking')
        #     else:
        #         num_entries = np.sum(batch_num_nodes * batch_num_nodes)
        #         embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        #         adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
        #         self.link_loss[(1-adj_mask).bool()] = 0.0
        #
        #     self.link_loss = torch.sum(self.link_loss) / float(num_entries)
        #     #print('linkloss: ', self.link_loss)
        #     return loss + self.link_loss
        return loss
