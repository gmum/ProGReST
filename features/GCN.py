import torch
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d, Dropout
from torch_geometric.nn import GCNConv, ResGatedGraphConv
from torch_geometric.nn import Sequential as SequentialGraph
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import BatchNorm as BatchNormGraph


class GCN(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, classes=1):
        super(GCN, self).__init__()

        self.name = "base"

        layers = [(GCNConv(input_size, hidden_sizes[0]), 'x, edge_index -> x')]

        for i in range(len(hidden_sizes)-1):
            layers += [
                ReLU(),
                (GCNConv(hidden_sizes[i], hidden_sizes[i+1]), 'x, edge_index -> x')
            ]

        self.features = SequentialGraph('x, edge_index',  layers)

        self.out = Sequential(
            Linear(hidden_sizes[-1] * 2, hidden_sizes[-1]),
            BatchNorm1d(hidden_sizes[-1]),
            ReLU(),
            Dropout(0.5),

            Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            BatchNorm1d(hidden_sizes[-1] // 2),
            ReLU(),
            Dropout(0.25),

            Linear(hidden_sizes[-1] // 2, classes)
        )

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        batch_index = batch.batch
        features = self.features(x, edge_index)
        features = torch.cat([gmp(features, batch_index), gap(features, batch_index)], dim=1)
        out = self.out(features)

        return out


class NNConv(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, classes=1):
        super(NNConv, self).__init__()

        self.name = "base"

        layers = [(NNConv(input_size, hidden_sizes[0]), 'x, edge_index, edge_attr -> x')]

        for i in range(len(hidden_sizes)-1):
            layers += [
                ReLU(),
                (NNConv(hidden_sizes[i], hidden_sizes[i+1]), 'x, edge_index, edge_attr -> x')
            ]

        self.features = SequentialGraph('x, edge_index, edge_attr',  layers)

        self.out = Sequential(
            Linear(hidden_sizes[-1] * 2, hidden_sizes[-1]),
            BatchNorm1d(hidden_sizes[-1]),
            ReLU(),
            Dropout(0.5),

            Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            BatchNorm1d(hidden_sizes[-1] // 2),
            ReLU(),
            Dropout(0.25),

            Linear(hidden_sizes[-1] // 2, classes)
        )

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        batch_index = batch.batch
        features = self.features(x, edge_index, batch.edge_attr)
        features = torch.cat([gmp(features, batch_index), gap(features, batch_index)], dim=1)
        out = self.out(features)

        return out

class GCNResidual(torch.nn.Module):
        def __init__(self, input_size, hidden_sizes, classes=1):
            super(GCNResidual, self).__init__()

            self.name = "base"

            layers = [(ResGatedGraphConv(input_size, hidden_sizes[0]), 'x, edge_index -> x')]

            for i in range(len(hidden_sizes) - 1):
                layers += [
                    #     BatchNormGraph(hidden_sizes[i]),

                    ReLU(),
                    (ResGatedGraphConv(hidden_sizes[i], hidden_sizes[i + 1]), 'x, edge_index -> x')
                ]

            self.features = SequentialGraph('x, edge_index', layers)

            self.out = Sequential(
                Linear(hidden_sizes[-1] * 2, hidden_sizes[-1]),
                BatchNorm1d(hidden_sizes[-1]),
                ReLU(),
                Dropout(0.5),

                Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
                BatchNorm1d(hidden_sizes[-1] // 2),
                ReLU(),
                Dropout(0.25),

                Linear(hidden_sizes[-1] // 2, classes)
            )

        def forward(self, batch):
            x = batch.x
            edge_index = batch.edge_index
            batch_index = batch.batch
            features = self.features(x, edge_index)
            features = torch.cat([gmp(features, batch_index), gap(features, batch_index)], dim=1)
            out = self.out(features)

            return out
