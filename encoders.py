# -*- coding: utf-8 -*-
'''
@Time    : 2021-04-12 9:25 p.m.
@Author  : datasnail
@File    : encoders.py.py
'''

import torch
import numpy as np
import math
from torch import nn, einsum
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, seq_mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # mask
        print(dots.shape, seq_mask.shape, v.shape)
        if not (seq_mask is None):
            seq_mask = repeat(seq_mask[:, np.newaxis],'b () d -> b n d', n=dots.shape[2])
            print(seq_mask.shape)
            seq_mask = repeat(seq_mask[:,np.newaxis], 'b () n d -> b h n d',h=dots.shape[1])
            dots = einsum('b h i j, b h i j->b h i j', dots, seq_mask)

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, seq_mask=None):
        for attn, ff in self.layers:
            x = attn(x, seq_mask=seq_mask) + x
            x = ff(x) + x
        return x


class GSTransformer(nn.Module):
    def __init__(self, token_num, token_dim, dim, heads, mlp_dim, nlayers, nclass, pool='cls', dim_head=64, dropout=0.5, emb_dropout=0., has_mask=False):
        super(GSTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.has_mask = has_mask

        self.to_embedding = nn.Sequential(
            nn.Linear(token_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, token_num + 1, dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.seq_mask = None

        self.transformer = Transformer(dim, nlayers, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.pred = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, nclass)
            )

    def _generate_seq_mask(self, bz, ntoken, num_nodes):
        pad_mask = torch.ones((bz, ntoken), dtype=torch.bool)

        for index, item in enumerate(num_nodes):
            pad_mask[index, item+1:] = 0  # 已经加上cls位了
        return pad_mask

    def forward(self, seq_feats, num_nodes):
        x = self.to_embedding(seq_feats)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        if self.has_mask:
            device = x.device
            if self.seq_mask is None or self.seq_mask.size(0) != len(seq_feats):
                # print('using sequence padding mask...')
                bz = x.size(0)  # seq_feats.size(0)
                ntoken = x.size(1)  # seq_feats.size(1)+1
                mask = self._generate_seq_mask(bz, ntoken, num_nodes).to(device)
                self.seq_mask = mask
        else:
            # print('Do NOT use sequence padding mask...')
            self.seq_mask = None

        x = self.transformer(x,self.seq_mask)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.pred(x)


    def loss(self, pred, label, type='softmax'):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        # print(pred.shape, label.shape)
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


class GSRNN(nn.Module):
    def __init__(self, token_dim, dim, hidden_size, num_layers, nclass, net_type=0):  # nonlinearity='tanh', bias=False, batch_first=True, dropout=0., bidirectional=False
        super(GSRNN, self).__init__()
        self.net_type=net_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.to_embedding = nn.Sequential(
            nn.Linear(token_dim, dim),
        )

        if self.net_type == 1:
            self.model_type = 'GSLSTM'
            self.rnn = nn.LSTM(dim, self.hidden_size, self.num_layers, batch_first=True)
        else:
            self.model_type = 'GSRNN'
            self.rnn = nn.RNN(dim, self.hidden_size, self.num_layers, batch_first=True)

        self.to_latent = nn.Identity()
        self.pred = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, nclass)
            )

    def forward(self, seq_feats, num_nodes):
        x = self.to_embedding(seq_feats)
        if self.net_type == 1:
            h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).cuda()
            c0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).cuda()
            y, (h, c) = self.rnn(x, (h0,c0))
        else:
            y, h = self.rnn(x)

        y = self.to_latent(y)

        # return self.pred(y[:,-1,:])
        return self.pred(y[range(y.shape[0]), num_nodes, :])

    def loss(self, pred, label):
        return F.cross_entropy(pred, label, reduction='mean')
