import numpy as np
import torch
from huggingmolecules import RMatModel, RMatFeaturizer
from sklearn.cluster import KMeans
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from args import get_args
from model.prototree import RMatProtoSoftTree, GCNProtoSoftTree
from utils.data_loader import get_data_loaders_rmat
from utils.draw import draw_similarity, draw_last_layer
from utils.train_test import train, test
import json

import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.mock = nn.Identity()

    def forward(self, x, y, z):
        return self.mock(x)


def train_model_reg(args, name=""):
    args.task_type = "clst"
    pt, cl, prob, ld,mse = args.pt, args.cl, 0, args.ld, 0
    lr = args.lr

    lr_add, lr_features, lr_protos, lr_leaves = lr, lr * 5e-4, lr, lr
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    model = RMatModel.from_pretrained('rmat_4M', excluded=['generator'])
    model.generator = Decoder()
    featurizer = RMatFeaturizer.from_pretrained('rmat_4M')

    dl_train, dl_eval, dl_test, dl_proj = get_data_loaders_rmat(featurizer,
                                                                batch_size=args.batch_size,
                                                                task_name='Tox',
                                                                dataset_name=args.dataset_name,
                                                                split_method=args.split_method,
                                                                cache_encodings=args.cache_encodings,
                                                                split_seed=args.split_seed)

    train_y = np.array([x.y for x in dl_train.dataset])
    kmeans = KMeans(n_clusters=args.classes).fit(train_y.reshape(-1, 1))
    criterion = nn.NLLLoss()

    protoTree: GCNProtoSoftTree = RMatProtoSoftTree(model, args)
    protoTree.to(device)

    parameters = [
        {'params': protoTree.features.parameters(), 'lr': lr_features, 'weight_decay': 1e-3},
        {'params': protoTree.add_on.parameters(), 'lr': lr_add},
        #    {'params': protoTree.prototype_vectors, 'lr': lr_protos},
        {'params': protoTree.leaves, 'lr': lr_leaves}
    ]

    for i in range(protoTree.depth):
        depth = protoTree.depth - i - 1
        parameters.append({'params': protoTree.prototypes[i], 'lr': lr_protos * (2 ** (-depth))})

    optimizer = Adam(parameters)
    protoTree.warmup()

    model_name = f"{str(protoTree)}_LR_{args.lr}_CL_{args.cl}_PT_{pt}_Prob_{args.prob}_LD_{args.ld}_S_{args.prototype_size}_seed_{args.split_seed}"
    print(model_name)
    writer = SummaryWriter(f'../{args.dataset_name}/{model_name}+{name}')

    best = 0 if args.task_type == "clst" else 999999999


    for epoch in range(args.warmup):
        total_parts_train, total_stats_train = train(protoTree, criterion, dl_train, optimizer, device,
                                                     task_type=args.task_type, pt=pt, cl=cl, prob=prob, ld=ld, mse=mse, kmeans=kmeans)

        info = ""
        for key, value in total_stats_train.items():
            info += f" {key.upper()}: {value:.3f}"

        print(f"Train Epoch:{epoch}  |Loss: {total_parts_train['loss']:.3f} |" + info)

        total_parts_eval, total_stats_eval = test(protoTree, criterion, dl_eval, optimizer, device,
                                                  task_type=args.task_type, pt=pt, cl=cl, prob=prob, ld=ld, mse=mse, kmeans=kmeans)

        info = ""
        for key, value in total_stats_eval.items():
            info += f" {key.upper()}: {value:.3f}"

        print(f"Eval Epoch:{epoch}  |Loss: {total_parts_eval['loss']:.3f} |" + info)
        #

        for key, value in total_parts_train.items():
            writer.add_scalar(f'Parts/{key}', value, epoch)

        for key in total_stats_train.keys():
            writer.add_scalar(f'{key}/Train', total_stats_train[key], epoch)
            writer.add_scalar(f'{key}/Eval', total_stats_eval[key], epoch)

        if epoch % 10 == 0:
            data = next(iter(dl_train))
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



    # =========================================== REG=====================

    criterion = nn.MSELoss()
    args.task_type = "reg"
    args.classes = 1

    old_protoTree = protoTree
    leaves = F.softmax(old_protoTree.leaves,dim=1).cpu().detach().numpy()
    leaves = (leaves * kmeans.cluster_centers_.reshape(-1)).sum(-1).reshape(-1, 1)

    protoTree: GCNProtoSoftTree = RMatProtoSoftTree(model, args)
    for i in range(protoTree.depth):
        protoTree.prototypes[i] = nn.Parameter(old_protoTree.prototypes[i].data, requires_grad=True)

    protoTree.leaves = nn.Parameter(torch.Tensor(leaves), requires_grad=True)

    protoTree.to(device)

    parameters = [
        {'params': protoTree.features.parameters(), 'lr': lr_features, 'weight_decay': 1e-3},
        {'params': protoTree.add_on.parameters(), 'lr': lr_add},
        #    {'params': protoTree.prototype_vectors, 'lr': lr_protos},
        {'params': protoTree.leaves, 'lr': lr_leaves}
    ]

    for i in range(protoTree.depth):
        depth = protoTree.depth - i - 1
        parameters.append({'params': protoTree.prototypes[i], 'lr': lr_protos * (2 ** (-depth))})

    optimizer = Adam(parameters)


    protoTree.last_only()

    for proj_idx in range(5):
        _, _ = train(protoTree, criterion, dl_train, optimizer, device,
                     task_type=args.task_type, pt=args.pt, cl=args.cl, prob=prob, ld=args.ld,  kmeans=kmeans)

    total_parts_eval, total_stats_eval = test(protoTree, criterion, dl_eval, optimizer, device,
                                              task_type=args.task_type, pt=pt, cl=cl, prob=prob, ld=ld, mse=mse,  kmeans=kmeans)

    info = ""
    for key, value in total_stats_eval.items():
        info += f" {key.upper()}: {value:.3f}"

    print(f"PROJ Eval Epoch:{args.warmup}  |Loss: {total_parts_eval['loss']:.3f} |" + info)

    total_parts_test, total_stats_test = test(protoTree, criterion, dl_test, optimizer, device,
                                              task_type=args.task_type, pt=pt, cl=cl, prob=prob, ld=ld, mse=mse,  kmeans=kmeans)

    info = ""
    for key, value in total_stats_test.items():
        info += f" {key.upper()}: {value:.3f}"

    print(f"PROJ Test Epoch:{args.warmup}  |Loss: {total_parts_test['loss']:.3f} |" + info)

    best = 999999999

    protoTree.warmup()
    for epoch in range(args.warmup, args.epoch):
        if epoch >= args.project_start and epoch % args.project_mod == 0:
            protoTree.eval()
            protoTree.project_fast_prototypes(dl_proj)
            protoTree.normalize()
            protoTree.train()

            protoTree.last_only()

            for proj_idx in range(5):
                _, _ = train(protoTree, criterion, dl_train, optimizer, device,
                             task_type=args.task_type, pt=args.pt, cl=args.cl, prob=prob, ld=args.ld)

            total_parts_eval, total_stats_eval = test(protoTree, criterion, dl_eval, optimizer, device,
                                                      task_type=args.task_type, pt=pt, cl=cl, prob=prob, ld=ld, mse=mse)

            info = ""
            for key, value in total_stats_eval.items():
                info += f" {key.upper()}: {value:.3f}"

            print(f"PROJ Eval Epoch:{epoch}  |Loss: {total_parts_eval['loss']:.3f} |" + info)

            total_parts_test, total_stats_test = test(protoTree, criterion, dl_test, optimizer, device,
                                                      task_type=args.task_type, pt=pt, cl=cl, prob=prob, ld=ld, mse=mse)

            info = ""
            for key, value in total_stats_test.items():
                info += f" {key.upper()}: {value:.3f}"

            print(f"PROJ Test Epoch:{epoch}  |Loss: {total_parts_test['loss']:.3f} |" + info)

        if epoch >= args.join:
            protoTree.join()
        else:
            protoTree.warmup()


        total_parts_train, total_stats_train = train(protoTree, criterion, dl_train, optimizer, device,
                                                     task_type=args.task_type, pt=pt, cl=cl, prob=prob, ld=ld, mse=mse)

        info = ""
        for key, value in total_stats_train.items():
            info += f" {key.upper()}: {value:.3f}"

        print(f"Train Epoch:{epoch}  |Loss: {total_parts_train['loss']:.3f} |" + info)

        total_parts_eval, total_stats_eval = test(protoTree, criterion, dl_eval, optimizer, device,
                                                  task_type=args.task_type, pt=pt, cl=cl, prob=prob, ld=ld, mse=mse)

        info = ""
        for key, value in total_stats_eval.items():
            info += f" {key.upper()}: {value:.3f}"

        print(f"Eval Epoch:{epoch}  |Loss: {total_parts_eval['loss']:.3f} |" + info)

        total_parts_test, total_stats_test = test(protoTree, criterion, dl_test, optimizer, device,
                                                  task_type=args.task_type, pt=pt, cl=cl, prob=prob, ld=ld, mse=mse)

        info = ""
        for key, value in total_stats_test.items():
            info += f" {key.upper()}: {value:.3f}"

        print(f"Test Epoch:{epoch}  |Loss: {total_parts_test['loss']:.3f} |" + info)

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
            data = next(iter(dl_train))
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
            print(f"BEST! {best}")

            torch.save({
                'add_on': protoTree.add_on.state_dict(),
                'prototypes': protoTree.prototypes,
                'leaves': protoTree.leaves
            }, f'./saves/{args.dataset_name}_{model_name}.pt')

    checkpoint = torch.load(f'./saves/{args.dataset_name}_{model_name}.pt')
    protoTree.add_on.load_state_dict(checkpoint['add_on'])
    protoTree.prototypes = checkpoint['prototypes']
    protoTree.leaves = checkpoint['leaves']

    _, total_stats_train = test(protoTree, criterion, dl_train, optimizer, device,
                                task_type=args.task_type, pt=pt, cl=cl, prob=prob, ld=ld, mse=mse)

    _, total_stats_eval = test(protoTree, criterion, dl_eval, optimizer, device,
                               task_type=args.task_type, pt=pt, cl=cl, prob=prob, ld=ld, mse=mse)

    _, total_stats_test = test(protoTree, criterion, dl_test, optimizer, device,
                               task_type=args.task_type, pt=pt, cl=cl, prob=prob, ld=ld, mse=mse)

    return total_stats_train, total_stats_eval, total_stats_test



def stats_reg(args):

    # DATASET ========================
    model_name = f"RMATProtoTree_{args.depth}_{args.latent_distance}_LR_{args.lr}_CL_{args.cl}_PT_{args.pt}_Prob_{args.prob}_LD_{args.ld}_S_{args.prototype_size}_seed_{args.split_seed}"

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    model = RMatModel.from_pretrained('rmat_4M', excluded=['generator'])
    model.generator = Decoder()
    featurizer = RMatFeaturizer.from_pretrained('rmat_4M')

    dl_train, dl_eval, dl_test, dl_proj = get_data_loaders_rmat(featurizer,
                                                                batch_size=args.batch_size,
                                                                task_name='Tox',
                                                                dataset_name=args.dataset_name,
                                                                split_method=args.split_method,
                                                                cache_encodings=args.cache_encodings,
                                                                split_seed=args.split_seed)

    protoTree: GCNProtoSoftTree = RMatProtoSoftTree(model, args)
    protoTree.to(device)

    protoTree.load(torch.load(f'./saves/{args.dataset_name}_{model_name}.pt', map_location=device))
    protoTree.leaves_func = nn.Identity()
    protoTree.classes =1
    protoTree.eval()

    total_parts_test, total_stats_test = test(protoTree, None, dl_test, None, device, task_type=args.task_type)
    return total_stats_test[args.metric]


if __name__ == '__main__':

    args = get_args()

    args.lr = 0.02
    args.dataset_name = 'LD50_Zhu'
    args.classes = 48
    args.metric = 'mae'
    args.pt = 0.3
    args.cl = 0.01
    args.ld = 0.01
    args.prob = 0
    args.batch_size = 32
    args.depth = 6
    args.split_method = 'scaffold'

    args.epoch = 150
    args.batch_size = 32
    args.prototype_size = 32
    args.latent_distance = "parents"
    args.project_start = 100
    args.project_mod = 10
    args.warmup = 75
    args.join = 80

    torch.manual_seed(2137)
    args.split_seed = 1
    total_stats_train, total_stats_eval, total_stats_test = train_model_reg(args)




