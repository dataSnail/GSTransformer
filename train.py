# -*- coding: utf-8 -*-
'''
@Time    : 2021-04-12 8:14 p.m.
@Author  : datasnail
@File    : train.py
'''

import os
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import sklearn.metrics as metrics
from tensorboardX import SummaryWriter

import load_data
import cross_val
import gen.feat as featgen
import gen.data as datagen
import util
import encoders

import time
from torch.autograd import Variable

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def evaluate(dataset, model, args, name='Validation', max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        seq_feats = Variable(data['seq_feats'].float(), requires_grad=False).cuda()
        # h0 = Variable(data['feats'].float()).cuda()
        labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        # assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()

        ypred = model(adj, seq_feats, batch_num_nodes)  # bugs if use RNN-liked model, as has no adj parameter
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}
    print(name, " accuracy:%.4f"%result['acc'])
    return result

def benchmark_task_val(args, writer=None, feat='node-label'):
    all_vals = []
    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)

    example_node = util.node_dict(graphs[0])[0]

    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in example_node:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['label'])
    else:
        print('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    for i in range(10):
        train_dataset, val_dataset, max_num_nodes, input_dim = \
                cross_val.prepare_val_data(graphs, args, i, max_nodes=args.max_nodes)

        if args.method == 'GSTransformer':
            print('Method: %s, Mask_flag: %s' % (args.method, args.no_mask))
            model = encoders.GSTransformer(
                    max_num_nodes, input_dim, args.dim, args.num_heads, args.mlp_dim, args.num_trans_layers, args.num_classes,
                    pool=args.pool, dropout=args.dropout, has_mask=args.no_mask, mask_type=args.mask_type).cuda()
        elif args.method == 'GSRNN':
            # RNN-liked methods args.mlp_dim: hidden_size; args.num_trans_layers: num_layers
            print('Method: %s'%args.method)
            model = encoders.GSRNN(
                    input_dim, args.dim, args.mlp_dim, args.num_trans_layers, args.num_classes, 0).cuda()
        elif args.method == 'GSLSTM':
            print('Method: %s'%args.method)
            model = encoders.GSRNN(
                input_dim, args.dim, args.mlp_dim, args.num_trans_layers, args.num_classes, 1).cuda()
        else:
            print("ERROR Methods!!!")
            exit(1)
        _, val_accs = train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=None,
            writer=writer)
        all_vals.append(np.array(val_accs))
    all_vals = np.vstack(all_vals)
    all_vals = np.mean(all_vals, axis=0)
    print("Mean:----<%s>"%i,all_vals)
    print(np.max(all_vals))
    print(np.argmax(all_vals))

def gen_train_plt_name(args):
    return 'results/' + gen_prefix(args) + '.png'

def train(dataset, model, args, val_dataset=None, test_dataset=None, writer=None,
          mask_nodes=True):
    writer_batch_idx = [0, 3, 6, 9]
    p_c = 0
    for pp in model.parameters():
        if pp.requires_grad:
            p_c+=1
    print('---------------',p_c)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    iter = 0
    best_val_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    test_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []
    for epoch in range(args.num_epochs):
        total_time = 0
        avg_loss = 0.0
        model.train()
        print('Epoch: ', epoch)
        for batch_idx, data in enumerate(dataset):
            begin_time = time.time()
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            seq_feats = Variable(data['seq_feats'].float(), requires_grad=False).cuda()
            label = Variable(data['label'].long()).cuda()
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None

            ypred = model(adj, seq_feats, batch_num_nodes)

            loss = model.loss(ypred, label)
            # if not args.method == 'soft-assign' or not args.linkpred:
            #     loss = model.loss(ypred, label)
            # else:
            #     loss = model.loss(ypred, label, adj, batch_num_nodes)
            # print(model.parameters())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # 梯度裁剪
            optimizer.step()
            iter += 1
            avg_loss += loss
            # if iter % 20 == 0:
            #    print('Iter: ', iter, ', loss: ', loss.data[0])
            elapsed = time.time() - begin_time
            total_time += elapsed

            # log once per XX epochs
            # if epoch % 10 == 0 and batch_idx == len(
            #         dataset) // 2 and writer is not None:
            #     log_assignment(model.assign_tensor, writer, epoch, writer_batch_idx)
            #     if args.log_graph:
            #         log_graph(adj, batch_num_nodes, writer, epoch, writer_batch_idx, model.assign_tensor)
        avg_loss /= batch_idx + 1
        if writer is not None:
            writer.add_scalar('loss/avg_loss', avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar('loss/linkpred_loss', model.link_loss, epoch)
        print('[LOSS] Avg loss: %.4f ; epoch time: %.2f'% (avg_loss.item(), total_time))
        result = evaluate(dataset, model, args, name='Train', max_num_examples=100)
        train_accs.append(result['acc'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, args, name='Validation')
            val_accs.append(val_result['acc'])
        if val_result['acc'] > best_val_result['acc'] - 1e-7:
            best_val_result['acc'] = val_result['acc']
            best_val_result['epoch'] = epoch
            best_val_result['loss'] = avg_loss
        if test_dataset is not None:
            test_result = evaluate(test_dataset, model, args, name='Test')
            test_result['epoch'] = epoch
        if writer is not None:
            writer.add_scalar('acc/train_acc', result['acc'], epoch)
            writer.add_scalar('acc/val_acc', val_result['acc'], epoch)
            writer.add_scalar('loss/best_val_loss', best_val_result['loss'], epoch)
            if test_dataset is not None:
                writer.add_scalar('acc/test_acc', test_result['acc'], epoch)

        print('【BEST】 val result: %.4f, Epoch:%s.'%(best_val_result['acc'],best_val_result['epoch']))
        best_val_epochs.append(best_val_result['epoch'])
        best_val_accs.append(best_val_result['acc'])
        if test_dataset is not None:
            print('Test result: ', test_result)
            test_epochs.append(test_result['epoch'])
            test_accs.append(test_result['acc'])

    matplotlib.style.use('seaborn')
    plt.switch_backend('agg')
    plt.figure()
    plt.plot(train_epochs, util.exp_moving_avg(train_accs, 0.85), '-', lw=1)
    if test_dataset is not None:
        plt.plot(best_val_epochs, best_val_accs, 'bo', test_epochs, test_accs, 'go')
        plt.legend(['train', 'val', 'test'])
    else:
        plt.plot(best_val_epochs, best_val_accs, 'bo')
        plt.legend(['train', 'val'])
    plt.savefig(gen_train_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use('default')

    return model, val_accs


def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
    name += '_' + args.method
    if args.method == 'GSTransformer':
        name += '_dim' + str(args.dim) + '_mlpDim' + str(args.mlp_dim)
        name += '_tansLayer' + str(args.num_trans_layers) + '_heads'+str(args.num_heads) + '_lr'+str(args.lr)
        name += '_'+str(args.pool) + '_' + str(args.sort_type) + '_nomask'
        name += '_'+str(args.mask_type)
        if args.linkpred:
            name += '_lp'
    else:
        name += '_h' + str(args.hidden_dim) + '_o' + str(args.output_dim)

    if len(args.name_suffix) > 0:
        name += '_' + args.name_suffix
    return name


def arg_parse():
    parser = argparse.ArgumentParser(description='GSTransformer arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset',
            help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname',
            help='Name of the benchmark dataset')
    io_parser.add_argument('--pkl', dest='pkl_fname',
            help='Name of the pkl data file')

    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument('--assign-ratio', dest='assign_ratio', type=float,
            help='ratio of number of nodes in consecutive layers')
    softpool_parser.add_argument('--num-pool', dest='num_pool', type=int,
            help='number of pooling layers')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
            const=True, default=False,
            help='Whether link prediction side objective is used')


    parser.add_argument('--datadir', dest='datadir',
            help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
            help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
            help='CUDA.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
            help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
            help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
            help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
            help='Input feature dimension')
    parser.add_argument('--dim', dest='dim', type=int,
            help='Input embedding dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
            help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
            help='Number of label classes')
    parser.add_argument('--num-heads', dest='num_heads', type=int,
            help='Head number of transformer')

    parser.add_argument('--num-trans-layers', dest='num_trans_layers', type=int,
            help='Number of Transformer layers')
    parser.add_argument('--mlp-dim', dest='mlp_dim', type=int,
            help='Input dimension of prediction layer')
    parser.add_argument('--nobn', dest='bn', action='store_const',
            const=False, default=True,
            help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')
    parser.add_argument('--no-mask', dest='no_mask', action='store_const',
            const=False, default=True,
            help="Whether to add mask to sequence.")
    # parser.add_argument('--cls', dest='cls_flag', action='store_const',
    #         const=True, default=False,
    #         help="Whether to add mask to sequence.")
    parser.add_argument('--no-log-graph', dest='log_graph', action='store_const',
            const=False, default=True,
            help='Whether disable log graph')
    parser.add_argument('--mask', dest='mask_type',
            help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--method', dest='method',
            help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--sort-type', dest='sort_type',
                        help='The type of transfering graph to sequence [degree0, degree1, bfs]')
    parser.add_argument('--pool', dest='pool',
            help='pool cls or mean')
    parser.add_argument('--name-suffix', dest='name_suffix',
            help='suffix added to the output filename')

    parser.set_defaults(datadir='data',
                        logdir='log',
                        dataset='syn1v2',
                        max_nodes=1000,
                        cuda='1',
                        feature_type='default',
                        lr=0.001,
                        clip=2.0,
                        batch_size=20,
                        num_epochs=1000,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=1,
                        # input_dim=10,
                        # hidden_dim=20,
                        # output_dim=20,
                        num_classes=2,
                        num_heads=8,
                        num_gc_layers=3,
                        dropout=0.5,
                        method='base',
                        pool='cls',
                        sort_type='degree1',
                        name_suffix='',
                        mask_type='seq'
                       )
    return parser.parse_args()


def main():
    prog_args = arg_parse()

    # export scalar data to JSON for external processing
    path = os.path.join(prog_args.logdir, gen_prefix(prog_args))
    if os.path.isdir(path):
        print('Remove existing log dir: ', path)
        shutil.rmtree(path)
    writer = SummaryWriter(path)

    os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda
    print('CUDA', prog_args.cuda)

    if prog_args.bmname is not None:
        benchmark_task_val(prog_args, writer=writer, feat='node-label')

    writer.close()


if __name__=='__main__':
    main()