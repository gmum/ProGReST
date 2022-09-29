# import packages
# general tools
import copy
import json
import math
from collections import Counter
from pathlib import Path
import torch.nn.functional as F

from torch_geometric.data import Data

# Pytorch and Pytorch Geometric
import networkx as nx
import numpy as np
import pandas as pd
import torch
# RDkit
from matplotlib import pyplot as plt
from rdkit import Chem
from skimage.io import imread
from sklearn.preprocessing import MinMaxScaler
from tdc.benchmark_group import admet_group
from torch import nn
from torch_geometric.data import DataLoader

from args import get_args
from features.GCN import GCN
from model.prototree import GCNProtoSoftTree
from utils.draw import img_for_mol
from utils.featurizer import from_smiles

from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool

import timeit


class MCTSNode():

    def __init__(self, coalition: list, c_puct: float = 10.0,
                 W: float = 0, N: int = 0, P: float = 0):
        self.coalition = coalition
        self.c_puct = c_puct
        self.children = []
        self.W = W  # sum of node value
        self.N = N  # times of arrival
        self.P = P  # property score (reward)

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)


class MCTS():
    expand_atoms = 18
    min_atoms = 3
    max_atoms = 12
    rollout = 32

    def __init__(self, data, model, prototype):
        graph = nx.Graph().to_undirected()
        graph.add_nodes_from(range(data.x.shape[0]))
        for start, end in zip(data.edge_index[0], data.edge_index[1]):
            graph.add_edge(start.item(), end.item())

        self.data = data
        self.graph = graph
        self.prototype = prototype
        self.model = model
        self.state_map = dict()
        self.pool = global_mean_pool

    def run(self):
        graph = self.graph
        root = MCTSNode(graph.nodes)
        self.state_map = {tuple(root.coalition): root}

        for rollout_id in range(self.rollout):
            print(max([x.P for x in self.state_map.values()]))
            self.mcts_rollout(root, graph)

        explanations = [node for _, node in self.state_map.items()]
        result_node = explanations[0]

        for result_idx in range(len(explanations)):
            x = explanations[result_idx]
            if len(x.coalition) <= self.max_atoms and x.P > result_node.P:
                result_node = x

        coalition = result_node.coalition
     ##   print(result_node.P)
     #   print(max([x.P for x in explanations]))

        masked = self.get_masked_graph(result_node.coalition)
        train_loader = DataLoader([masked], batch_size=1, shuffle=False)

        data = next(iter(train_loader)).to(self.model.leaves.device)
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = self.model.features(x, edge_index)
        x = self.pool(x, batch)
        proto = self.model.add_on(x.view(-1, self.model.features_size, 1)).squeeze()

        return coalition, proto, result_node.P


    def get_masked_graph(self, coalition):
        data = copy.deepcopy(self.data)

        mapping = {k:v for k,v in zip(coalition,range(len(coalition)))}
        x = data.x[torch.Tensor(coalition).long()]
        edges = list(nx.relabel_nodes(self.graph.subgraph(coalition).to_directed(), mapping).edges)
        if not edges:
            edges.append((0,0))
        return Data(x=x, edge_index=torch.Tensor(edges).T.long(), y=data.y)

    def score_func(self, coalition):
        masked = self.get_masked_graph(coalition)
        train_loader = DataLoader([masked], batch_size=1, shuffle=False)

        data = next(iter(train_loader)).to(self.model.leaves.device)

        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = self.model.features(x, edge_index)
        x = self.pool(x, batch)

        x = self.model.add_on(x.view(-1, self.model.features_size, 1)).squeeze()

        x2 = x ** 2
        p2 = self.prototype ** 2
        distance = -2 * (x @ self.prototype.T) + p2.sum(dim=-1, keepdims=True).T + x2.sum(dim=-1, keepdims=True)
        score = self.model.distance_2_similarity(distance)

        return score.item()

    def score(self, nodes):
        for node in nodes:
            if node.P == 0:
                node.P = self.score_func(node.coalition)

    def mcts_rollout(self, node, graph):
        cur_graph_coalition = node.coalition

        if len(cur_graph_coalition) <= self.min_atoms:
            return node.P

        if len(node.children) == 0:
            node_degree_list = list(graph.subgraph(cur_graph_coalition).degree)
            node_degree_list = sorted(node_degree_list, key=lambda x: x[1], reverse=False)
            all_nodes = [x[0] for x in node_degree_list]

            if len(all_nodes) < self.expand_atoms:
                expand_nodes = all_nodes
            else:
                expand_nodes = all_nodes[:self.expand_atoms]

            for each_node in expand_nodes:
                # for each node, pruning it and get the remaining sub-graph
                # here we check the resulting sub-graphs and only keep the largest one
                subgraph_coalition = [node for node in all_nodes if node != each_node]

                subgraphs = [graph.subgraph(subgraph_coalition)]
               #  subgraphs = [graph.subgraph(c)
               #               for c in nx.connected_components(graph.subgraph(subgraph_coalition))]
                main_sub = subgraphs[0]
                for sub in subgraphs:
                    if sub.number_of_nodes() > main_sub.number_of_nodes():
                        main_sub = sub

                new_graph_coalition = sorted(list(main_sub.nodes()))

                # check the state map and merge the same sub-graph
                Find_same = False
                for old_graph_node in self.state_map.values():
                    if Counter(old_graph_node.coalition) == Counter(new_graph_coalition):
                        new_node = old_graph_node
                        Find_same = True

                if Find_same == False:
                    new_node = MCTSNode(new_graph_coalition)
                    self.state_map[str(new_graph_coalition)] = new_node

                Find_same_child = False
                for cur_child in node.children:
                    if Counter(cur_child.coalition) == Counter(new_graph_coalition):
                        Find_same_child = True

                if Find_same_child == False:
                    node.children.append(new_node)

            self.score(node.children)

        sum_count = sum([c.N for c in node.children])
        selected_node = max(node.children, key=lambda x: x.Q() + x.U(sum_count))
        v = self.mcts_rollout(selected_node, graph)
        selected_node.W += v
        selected_node.N += 1
        return v

def explain_reg_mcts(args, features, generate_base_exp=True):

    # DATASET ========================
    group = admet_group(path='data/')
    benchmark = group.get(args.dataset_name)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    name = benchmark['name']

    train_df, valid_df = group.get_train_valid_split(benchmark=name, split_type=args.split_method, seed=args.split_seed)
    train_df = pd.DataFrame([x for x in train_df.itertuples() if Chem.MolFromSmiles(x.Drug).GetNumAtoms() <= 48])

    train_dataset = from_smiles(list(train_df.Drug), list(train_df.Y))
    proj_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    model_name = f"GCNProto_{args.classes}_{args.warmup}_LR_{args.lr}_CL_{args.cl}_PT_{args.pt}_LD_{args.ld}_S_{args.prototype_size}_{args.depth}_seed_{args.split_seed}"
    print(model_name)
    args.task_type = "reg"
    args.classes = 1
    mean = 0
    std = 1

    protoTree: GCNProtoSoftTree = GCNProtoSoftTree(features, args)
    protoTree.to(device)

    print(device)
    protoTree.load_state_dict(torch.load(f'./saves/{args.dataset_name}_{model_name}_proj.pt', map_location=device))
    protoTree.eval()
    protoTree.project_fast_prototypes(proj_loader)

    prototype_idxs = protoTree.get_best_molecule(proj_loader)

    protoTree.prototype_vectors = torch.cat(list(protoTree.prototypes)).to(protoTree.leaves.device)

    if generate_base_exp:
        with torch.no_grad():

            for idx in range(protoTree.num_prototypes):

                mol_idxs = list(np.random.randint(len(train_df),size=10)) + [prototype_idxs[idx].item()]
                best_p = 0
                best_idx = 0
                best_coalition = []
                best_proto = None
                for mol_idx in mol_idxs:
                    s_feature = proj_loader.dataset[mol_idx]

                    mcts = MCTS(s_feature, protoTree, protoTree.prototype_vectors[idx])
                    coalition, proto, p = mcts.run()

                    if p >= best_p:
                        best_coalition = coalition
                        best_idx = mol_idx
                        best_p = p
                        best_proto = proto.detach()

                depth = int(np.log2(idx + 1))
                idx_proto = idx - depth**2
                protoTree.prototypes[depth][idx_proto].data.copy_(best_proto.clone()).to(protoTree.leaves.device)

                mol = Chem.MolFromSmiles(train_df.iloc[best_idx].Drug)

                similarities = np.zeros(mol.GetNumAtoms())
                similarities[best_coalition] = 1.0

                   # fig = plt.figure(figsize=(8, 8), dpi=64)
                plt.figure(figsize=(8, 8), dpi=128)
                plt.imshow(img_for_mol(mol, atom_weights=similarities))
                plt.title(best_p)
                plt.axis('off')
                plt.savefig(f'explanations/{args.dataset_name}/{idx}_mcts_{best_idx}.png')
                plt.savefig(f'explanations/{args.dataset_name}/{idx}_mcts.png')

                plt.close('all')

                mol = Chem.MolFromSmiles(train_df.iloc[mol_idx].Drug)

                similarities = np.zeros(mol.GetNumAtoms())
                similarities[coalition] = 1.0

                # fig = plt.figure(figsize=(8, 8), dpi=64)
                plt.figure(figsize=(8, 8), dpi=128)
                plt.imshow(img_for_mol(mol, atom_weights=similarities))
                plt.title(p)
                plt.axis('off')
                plt.savefig(f'explanations/{args.dataset_name}/{idx}_mcts_{mol_idx}_patch.png')
                plt.close('all')

    with torch.no_grad():
        for i in range(0, 50):
            explanation = [0]
            smiles = train_df.iloc[i].Drug
            s_feature = proj_loader.dataset[i]

            mol = Chem.MolFromSmiles(smiles)


            Path(f'explanations/{args.dataset_name}/{smiles}/').mkdir(parents=True, exist_ok=True)

            while explanation[-1] < protoTree.num_prototypes:
                idx = explanation[-1]

                mcts = MCTS(s_feature, protoTree, protoTree.prototype_vectors[idx])
                coalition, _, p = mcts.run()

                if idx == explanation[-1]:
                    if p < 0.5:
                        child = 2*idx+1
                    else:
                        child = 2*idx+2
                    explanation.append(child)

                similarities = np.zeros(mol.GetNumAtoms())
                similarities[coalition] = 1.0


                img = imread(f'explanations/{args.dataset_name}/{idx}_mcts.png')

                fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=128, constrained_layout=True)
                plt.axis('off')

                axes[0].imshow(img_for_mol(mol, atom_weights=similarities))
                axes[0].set_axis_off()
                axes[1].imshow(img)
                axes[1].set_axis_off()
                fig.suptitle(str(p))

                plt.axis('off')
                plt.savefig(f'explanations/{args.dataset_name}/{smiles}/{idx}_mcts.png')
                plt.close('all')

            best = explanation[-1] - 2**protoTree.depth-1
            value = protoTree.leaves[best].item()
            y = proj_loader.dataset[i].y.item()

            result = {'nodes': explanation, 'pred': value, 'y':y}
            with open(f'explanations/{args.dataset_name}/{smiles}/result_mcts.json', 'w') as f:
                json.dump(result, f)
