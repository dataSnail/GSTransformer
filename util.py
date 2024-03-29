# -*- coding: utf-8 -*-
'''
@Time    : 2021-04-12 8:26 p.m.
@Author  : datasnail
@File    : util.py
'''

import networkx as nx

# ---- NetworkX compatibility
def node_iter(G):
    if nx.__version__<'2.0':
        return G.nodes()
    else:
        return G.nodes

def node_dict(G):
    if nx.__version__>'2.1':
        node_dict = G.nodes
    else:
        node_dict = G.node
    return node_dict
# ---------------------------

def exp_moving_avg(x, decay=0.9):
    shadow = x[0]
    a = [shadow]
    for v in x[1:]:
        shadow -= (1-decay) * (shadow-v)
        a.append(shadow)
    return a

