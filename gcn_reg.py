# import packages
# general tools
import copy

import numpy as np
# RDkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
# Pytorch and Pytorch Geometric
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from args import get_args
from features.GCN import GCN, GCNResidual

from gcn_explain_mcts import explain_reg_mcts
from model.prototree import GCNProtoSoftTree
from utils.draw import draw_last_layer, draw_similarity
from utils.featurizer import from_smiles
from utils.train_test import test, train
import torch.nn.functional as F
from tdc.benchmark_group import admet_group

def parse_info(stats):
    info = ""
    for key, value in stats.items():
        info += f" {key.upper()}: {value:.3f}"
    return  info



def train_reg(args, features):

    # DATASET ========================
    group = admet_group(path='data/')
    benchmark = group.get(args.dataset_name)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    name = benchmark['name']

    train_df, valid_df = group.get_train_valid_split(benchmark=name, split_type=args.split_method, seed=args.split_seed)

    train_dataset = from_smiles(list(train_df.Drug), list(train_df.Y))
    valid_dataset = from_smiles(list(valid_df.Drug), list(valid_df.Y))
    test_dataset = from_smiles(list(benchmark['test'].Drug), list(benchmark['test'].Y))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    proj_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    model_name = f"GCNProto_{args.classes}_{args.warmup}_LR_{args.lr}_CL_{args.cl}_PT_{args.pt}_LD_{args.ld}_S_{args.prototype_size}_{args.depth}_seed_{args.split_seed}"
    print(model_name)
    writer = SummaryWriter(f'../{args.dataset_name}_GCN/{model_name}')

    # WARMUP MODEL =====================
    args.task_type = "clst"
    features = features.to(device)

    protoTree: GCNProtoSoftTree = GCNProtoSoftTree(features, args)
    protoTree.train()

    protoTree.to(device)
    protoTree.join()
    lr = args.lr * 10
    lr_add, lr_features, lr_protos, lr_leaves = lr * 2 ** (-args.depth), lr * 2 ** (-args.depth), lr, lr

    parameters = [
        {'params': protoTree.features.parameters(), 'lr': lr_features},
        {'params': protoTree.add_on.parameters(), 'lr': lr_add},
        #    {'params': protoTree.prototype_vectors, 'lr': lr_protos},
        {'params': protoTree.leaves, 'lr': lr_leaves}
    ]

    for i in range(protoTree.depth):
        depth = protoTree.depth - i - 1
        parameters.append({'params': protoTree.prototypes[i], 'lr': lr_protos * (2 ** (-depth))})

    optimizer = torch.optim.Adam(parameters)
    pt, cl,  ld = args.pt, args.cl, args.ld

    train_y = np.array([x.y.item() for x in train_loader.dataset])
  #  mean = train_y.mean()
  #  std = train_y.std()
  #  train_y = (train_y-mean)/std
    mean = 0
    std = 1
    kmeans = KMeans(n_clusters=args.classes).fit(train_y.reshape(-1, 1))
    labels = kmeans.predict(train_y.reshape(-1, 1))
    weights = []
    for i in range(args.classes):
        weights.append(1/(len(labels[labels == i])/len(labels)))

    weights = torch.Tensor(weights)
    weights = weights/weights.max()
    #weights = weights/weights.norm()
    #criterion = nn.NLLLoss(weight=weights.to(device))
    criterion = nn.NLLLoss()

    for epoch in range(args.warmup):
        total_parts_train, total_stats_train = train(protoTree, criterion, train_loader, optimizer, device,task_type=args.task_type, pt=pt, cl=cl, ld=ld, kmeans=kmeans, mean=mean, std=std)
        print(f"Train Epoch:{epoch}  |Loss: {total_parts_train['loss']:.3f} |" + parse_info(total_stats_train))

        total_parts_eval, total_stats_eval = test(protoTree, criterion, valid_loader, optimizer, device, task_type=args.task_type, pt=pt, cl=cl,  ld=ld,  kmeans=kmeans, mean=mean, std=std)
        print(f"Eval Epoch:{epoch}  |Loss: {total_parts_eval['loss']:.3f} |" + parse_info(total_stats_eval))

        for key, value in total_parts_train.items():
            writer.add_scalar(f'Parts/{key}',  value, epoch)

        for key in total_stats_train.keys():
            writer.add_scalar(f'{key}/Train', total_stats_train[key], epoch)
            writer.add_scalar(f'{key}/Eval', total_stats_eval[key], epoch)

        if epoch % 10 == 0:
            data = next(iter(proj_loader))
            data = data.to(device)
            batch_activation = protoTree._forward_similarities(data)[0]

            linear = draw_last_layer(protoTree.get_leaves().cpu().detach().numpy().T)
            batch_activation = draw_similarity(batch_activation.detach().cpu().numpy())
            prot_sim_parents = draw_similarity(protoTree._get_sims_parents().detach().cpu().numpy())
            prot_sim_all = draw_similarity(protoTree._get_sims_all().detach().cpu().numpy())

            writer.add_image('Leaves', linear, epoch, dataformats='HWC')
            writer.add_image('Batch Activation', batch_activation, epoch, dataformats='HWC')
            writer.add_image('Proto sim parents', prot_sim_parents, epoch, dataformats='HWC')
            writer.add_image('Proto sim all', prot_sim_all, epoch, dataformats='HWC')

    # MODEL REG ================
    criterion = nn.MSELoss()
    args.task_type = "reg"
    args.classes = 1
    mean = 0
    std = 1

    old_protoTree = protoTree
    leaves = F.softmax(old_protoTree.leaves, dim=1).cpu().detach().numpy()
    leaves = (leaves * kmeans.cluster_centers_.reshape(-1)).sum(-1).reshape(-1, 1)

    protoTree: GCNProtoSoftTree = GCNProtoSoftTree(features, args)
    for i in range(protoTree.depth):
        protoTree.prototypes[i] = nn.Parameter(old_protoTree.prototypes[i].data, requires_grad=True)

    protoTree.leaves = nn.Parameter(torch.Tensor(leaves), requires_grad=True)
    protoTree.to(device)

    lr = args.lr
    lr_add, lr_features, lr_protos, lr_leaves = lr * 2 ** (-args.depth), lr * 2 ** (-args.depth), lr, lr

    parameters = [
        {'params': protoTree.features.parameters(), 'lr': lr_features},
        {'params': protoTree.add_on.parameters(), 'lr': lr_add},
        #    {'params': protoTree.prototype_vectors, 'lr': lr_protos},
        {'params': protoTree.leaves, 'lr': lr_leaves}
    ]

    for i in range(protoTree.depth):
        depth = protoTree.depth - i - 1
        parameters.append({'params': protoTree.prototypes[i], 'lr': lr_protos * (2 ** (-depth))})

    optimizer = torch.optim.Adam(parameters)
    protoTree.warmup()

    best = 999999
    best_proj = 99999

    for epoch in range(args.warmup, args.epoch):

        if epoch >= args.project_start and epoch % args.project_mod == 0:
            protoTree.eval()
            protoTree.project_fast_prototypes(proj_loader)
            protoTree.train()
            protoTree.last_only()

            for proj_idx in range(15):
                _, _ = train(protoTree, criterion, train_loader, optimizer, device, task_type=args.task_type, pt=pt, cl=cl, ld=ld,  mean=mean, std=std)

            total_parts_eval, total_stats_eval = test(protoTree, criterion, valid_loader, optimizer, device, task_type=args.task_type, pt=pt, cl=cl, ld=ld, mean=mean, std=std)
            total_parts_test, total_stats_test = test(protoTree, criterion, test_loader, optimizer, device, task_type=args.task_type, pt=pt, cl=cl, ld=ld, mean=mean, std=std)

            if epoch >= args.project_start and total_stats_eval[args.metric] < best_proj:
                best_proj = total_stats_eval[args.metric]
                print(f"BEST! {best_proj} - {total_stats_test[args.metric]}")

            torch.save(protoTree.state_dict(), f'./saves/{args.dataset_name}_{model_name}_proj.pt')


        if epoch >= args.join:
            protoTree.join()
        else:
            protoTree.warmup()


        total_parts_train, total_stats_train = train(protoTree, criterion, train_loader, optimizer, device,  task_type=args.task_type, pt=pt, cl=cl,  ld=ld, mean=mean, std=std)
        print(f"Train Epoch:{epoch}  |Loss: {total_parts_train['loss']:.3f} |" + parse_info(total_stats_train))

        total_parts_eval, total_stats_eval = test(protoTree, criterion, valid_loader, optimizer, device, task_type=args.task_type, pt=pt, cl=cl,  ld=ld, mean=mean, std=std)
        print(f"Eval Epoch:{epoch}  |Loss: {total_parts_eval['loss']:.3f} |" + parse_info(total_stats_eval))

        total_parts_test, total_stats_test = test(protoTree, criterion, test_loader, optimizer, device, task_type=args.task_type, pt=pt, cl=cl, ld=ld, mean=mean, std=std)
        print(f"Test Epoch:{epoch}  |Loss: {total_parts_test['loss']:.3f} |" + parse_info(total_stats_test))

        writer.add_scalar('Loss/train', total_parts_train['loss'], epoch)
        writer.add_scalar('Loss/eval', total_parts_eval['loss'], epoch)

        del total_parts_train['loss']

        for key, value in total_parts_train.items():
            writer.add_scalar(f'Parts/{key}', value, epoch)

        for key in total_stats_train.keys():
            writer.add_scalar(f'{key}/Train', total_stats_train[key], epoch)
            writer.add_scalar(f'{key}/Eval', total_stats_eval[key], epoch)
            writer.add_scalar(f'{key}/Test', total_stats_test[key], epoch)

        if epoch % 10 == 0:
            data = next(iter(proj_loader))
            data = data.to(device)
            batch_activation = protoTree._forward_similarities(data)[0]

            linear = draw_last_layer(protoTree.get_leaves().cpu().detach().numpy().T)
            batch_activation = draw_similarity(batch_activation.detach().cpu().numpy())
            prot_sim_parents = draw_similarity(protoTree._get_sims_parents().detach().cpu().numpy())
            prot_sim_all = draw_similarity(protoTree._get_sims_all().detach().cpu().numpy())

            writer.add_image('Leaves', linear, epoch, dataformats='HWC')
            writer.add_image('Batch Activation', batch_activation, epoch, dataformats='HWC')
            writer.add_image('Proto sim parents', prot_sim_parents, epoch, dataformats='HWC')
            writer.add_image('Proto sim all', prot_sim_all, epoch, dataformats='HWC')

        if epoch >= args.project_start and total_stats_eval[args.metric] < best:
            best = total_stats_eval[args.metric]
            print(f"BEST! {best} - {total_stats_test[args.metric]}")

        torch.save(protoTree.state_dict(), f'./saves/{args.dataset_name}_{model_name}_best.pt')

if __name__ == '__main__':
    torch.manual_seed(2137)

    args = get_args()

    args.features_size = 1024
    args.classes = 16
    args.depth = 5
    args.metric = 'mae'
    args.dataset_name = "LD50"

    args.lr = 0.005
    args.cuda = 0
    args.cl = 0.1
    args.ld = 0.01
    args.pt = 0.01

    args.epoch = 400
    args.batch_size = 64
    args.prototype_size = 128
    args.latent_distance = "parents"
    args.project_start = 250
    args.project_mod = 25
    args.warmup = 150
    args.join = 200
    args.split_seed = 1

    gnn_model = GCN(79, [256, 512, 512, 1024])

    train_reg(copy.deepcopy(args), gnn_model.features)
    explain_reg_mcts(copy.deepcopy(args), gnn_model.features, True)