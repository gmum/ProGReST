from torch import nn
import torch
import torch.nn.functional as F
import math
from torch_scatter import scatter


class GCNProtoSoftTree(nn.Module):

    def __init__(self, features, args):

        super(GCNProtoSoftTree, self).__init__()
        self.prototype_size = args.prototype_size
        self.features_size = args.features_size
        self.classes = args.classes
        self.problem = args.task_type

        depth = args.depth
        classes = args.classes

        self.prototypes = nn.ParameterList()

        for i in range(depth):
            self.prototypes.append(nn.Parameter(torch.rand((2 ** i, self.prototype_size)), requires_grad=True))
         #   self.prototypes.append(nn.Parameter(torch.full((2**i, self.prototype_size), 0.5), requires_grad=True))
            #self.prototypes.append(nn.Parameter(torch.normal(mean=0.5, std=0.25, size=(2 ** i, self.prototype_size)), requires_grad=True))

        self.num_prototypes = 2 ** depth - 1
        self.prototype_shape = (self.num_prototypes, self.prototype_size)

        self.prototype_vectors = None

        if self.problem == "clst":
            self.leaves = nn.Parameter(torch.zeros((2 ** depth, classes)), requires_grad=True)
        else:
            self.leaves = nn.Parameter(torch.randn((2 ** depth, classes)), requires_grad=True)

        self.depth = depth
        self.features = features

        self.add_on = nn.Sequential(
            nn.Conv1d(in_channels=self.features_size, out_channels=self.prototype_size, kernel_size=(1,), bias=False),
            nn.Sigmoid()
        )

        self.epsilon = 1e-8

        if classes == 1:
            self.leaves_func = nn.Identity()
        else:
            self.leaves_func = nn.Softmax(dim=1)

        parents = []

        for i in range(2 ** depth):
            idx = self.num_prototypes + i
            curr_parent = []
            for _ in range(depth):
                idx = (idx - 1) // 2
                curr_parent.append(idx)
            parents.append(curr_parent)

        self.parents = torch.Tensor(parents).long()

        self.latent_distance = args.latent_distance
        self.normalize()

    def __str__(self):
        return f"GCNProtoTree_{self.distance}_{self.depth}_{self.latent_distance}"

    def normalize(self):
        pass

    def _prototype_distance(self, x, prot):
        x2 = x**2
        p2 = prot**2
        distance = -2*(x @ prot.T) + p2.sum(dim=-1, keepdims=True).T+ x2.sum(dim=-1, keepdims=True)
        return distance, distance


    def prototype_distances(self, x):
        return self._prototype_distance(x, self.prototype_vectors)

    def network(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = self.features(x, edge_index)

        x = self.add_on(x.view(-1, self.features_size, 1)).squeeze()

        return x

    def distance_2_similarity(self, distances):
        return torch.exp(-distances)

    def get_best_distances(self, distances, latent_distance, batch):
        best_distances = scatter(distances, batch.batch, dim=0, reduce="min")
        best_latent_distances = best_distances

        return best_distances, best_latent_distances

    def _forward_similarities(self, data):
        features = self.network(data)

        distances, latent_distance = self.prototype_distances(features)

        best_distances, best_latent_distances = self.get_best_distances(distances, latent_distance, data)
        return self.distance_2_similarity(best_distances), best_distances, best_latent_distances

    def forward(self, data):
        self.prototype_vectors = torch.cat(list(self.prototypes)).to(self.leaves)

        batch_size = len(data.y)

        similarities, best_distances, best_latent_distances = self._forward_similarities(data)

        curr_probabilities = torch.ones(batch_size, 1).to(self.leaves.device)
        penalty = torch.zeros(1).to(self.leaves.device)

        for i in range(self.depth):
            start_idx, end_idx = 2 ** i - 1, 2 ** (i + 1) - 1
            sims = similarities[:, start_idx:end_idx]
            probabilities = torch.stack((1 - sims, sims), dim=2).view(batch_size, -1)
            penalty += self._penatly(curr_probabilities, sims, i)
            curr_probabilities = probabilities * curr_probabilities.repeat_interleave(2, dim=1)

        shape = (batch_size, 2 ** self.depth, self.classes)
        result = torch.sum(curr_probabilities.unsqueeze(2).broadcast_to(shape) * self.get_leaves().broadcast_to(shape), dim=1)

        return result, best_latent_distances, penalty, curr_probabilities


    def get_leaves(self):
        if self.problem == "clst":
            return self.leaves_func(self.leaves - torch.max(self.leaves, dim=1)[0].unsqueeze(1))
        else:
            return self.leaves_func(self.leaves)

    def _penatly(self, probs, sims, idx):
 #       probs = probs.detach()
        sims = torch.clamp(sims, min=self.epsilon, max=1-self.epsilon)
        probs = torch.clamp(probs, min=self.epsilon, max=1 - self.epsilon)

        alpha = torch.sum(probs * sims, dim=0) / torch.sum(probs, dim=0)
        penatly = -(2 ** (-idx)) * torch.sum(0.5 * torch.log(alpha) + 0.5 * torch.log(1 - alpha))
        return penatly

    def _get_sims_parents(self):
        sims = []
        indicates = torch.triu_indices(self.depth, self.depth, offset=1)
        proto_parents = self.prototype_vectors[self.parents]
        pp_n = proto_parents.norm(dim=-1, keepdim=True)

        for idx in range(0, 2 ** self.depth, 2):
            comp = (proto_parents[idx] @ proto_parents[idx].T) / (pp_n[idx] * pp_n[idx].T).clamp(min=0.00001)
            comp = torch.triu(comp, diagonal=1)
            sims.append(comp[indicates[0], indicates[1]].view(1, -1))

        return torch.cat(sims, dim=0)

    def _get_sims_all(self):
        pp = self.prototype_vectors
        pp_n = pp.norm(dim=1, keepdim=True)

        sims = (pp @ pp.T) / (pp_n * pp_n.T).clamp(min=0.00001)
        return torch.triu(sims, diagonal=1)

    def _get_sims(self):

        if self.latent_distance == "parents":
            return self._get_sims_parents()
        else:
            return self._get_sims_all()

    def get_costs(self, distances):


        max_dist = self.prototype_shape[1]
        inverted_distances, _ = torch.max((max_dist - distances), dim=1)
        cluster_cost = torch.mean(max_dist - inverted_distances)

        sims = self._get_sims()
        return cluster_cost, torch.norm(sims)

    def project_fast_prototypes(self, dl_proj):
        self.prototype_vectors = torch.cat(list(self.prototypes)).to(self.leaves.device)

        curr_prototypes = torch.zeros(self.prototype_shape)
        curr_similarities = torch.zeros(self.prototype_shape[0], 1)

        for batch in dl_proj:
            batch = batch.to(self.leaves.device)

            with torch.no_grad():
                x = self.network(batch)

                _, latent_distance = self.prototype_distances(x)
                similarities = self.distance_2_similarity(latent_distance)

                new_similarities, indexes = similarities.max(dim=0)
                new_similarities = new_similarities.detach().cpu().view(-1,1)

                new_prototypes = x[indexes].cpu().detach()

                selected_patch = (curr_similarities <= new_similarities).long()
                curr_prototypes = curr_prototypes * (1 - selected_patch) + new_prototypes * selected_patch
                curr_similarities = curr_similarities * (1 - selected_patch) + new_similarities * selected_patch

        for i, proto in enumerate(self.prototypes):
            start_idx, end_idx = 2 ** i - 1, 2 ** (i + 1) - 1
            proto.data.copy_(curr_prototypes[start_idx:end_idx].clone()).to(self.leaves)

    def get_best_molecule(self, dl_proj):
        self.prototype_vectors = torch.cat(list(self.prototypes)).to(self.leaves.device)

        curr_similarities = torch.zeros(self.prototype_shape[0])
        curr_ids = torch.zeros(self.prototype_shape[0])


        for idx, batch in enumerate(dl_proj):
            batch = batch.to(self.leaves.device)

            with torch.no_grad():
                x = self.network(batch)

                _, latent_distance = self.prototype_distances(x)
                similarities = self.distance_2_similarity(latent_distance)

                new_similarities, indexes = similarities.max(dim=0)
                new_similarities = new_similarities.detach().cpu()


                selected_patch = (curr_similarities <= new_similarities).long()
                curr_similarities = curr_similarities * (1 - selected_patch) + new_similarities * selected_patch
                curr_ids = curr_ids * (1 - selected_patch) + (dl_proj.batch_size * idx + batch.batch[indexes].cpu()) * selected_patch

        return curr_ids.long()


    def last_only(self):
        for param in self.features.parameters():
            param.requires_grad = False

        for param in self.add_on.parameters():
            param.requires_grad = False

        for param in self.prototypes:
            param.requires_grad = False

        self.leaves.requires_grad = True


    def join(self):
        for param in self.features.parameters():
            param.requires_grad = True

        for param in self.add_on.parameters():
            param.requires_grad = True

        for param in self.prototypes:
            param.requires_grad = True

        self.leaves.requires_grad = True


    def warmup(self):
        for param in self.features.parameters():
            param.requires_grad = False

        for param in self.add_on.parameters():
            param.requires_grad = True

        for param in self.prototypes:
            param.requires_grad = True

        self.leaves.requires_grad = True

    def disable(self):
        for param in self.features.parameters():
            param.requires_grad = False

        for param in self.add_on.parameters():
            param.requires_grad = False

        for param in self.prototypes:
            param.requires_grad = False

        self.leaves.requires_grad = False

    def to(self, device):
        nn.Module.to(self, device)
        self.prototype_vectors = torch.cat(list(self.prototypes)).to(device)


class RMatProtoSoftTree(GCNProtoSoftTree):

    def __init__(self, *args, **kwargs):

        super(RMatProtoSoftTree, self).__init__(*args, **kwargs)

    def __str__(self):
        return f"RMATProtoTree_{self.depth}_{self.latent_distance}"

    def load(self, state_dict):
        self.add_on.load_state_dict(state_dict['add_on'])
        self.prototypes = state_dict['prototypes']
        self.leaves = state_dict['leaves']

    def network(self, data):
        x = self.features(data)
        x = x.permute(0, 2, 1)

        x = self.add_on(x).permute(0, 2, 1)

        return x

    def distance_2_similarity(self, distances):
        return torch.exp(-distances)


    def get_best_distances(self, distances, latent_distances, batch):
        best_distances, _ = torch.min(distances, dim=1)
        best_latent_distances = best_distances

        return best_distances, best_latent_distances

    def project_fast_prototypes(self, dl_proj):
        self.prototype_vectors = torch.cat(list(self.prototypes)).to(self.leaves)

        curr_prototypes = torch.zeros(self.prototype_shape)
        curr_similarities = torch.zeros((self.prototype_shape[0],1))

        for batch in dl_proj:
            batch = batch.to(self.leaves)

            with torch.no_grad():
                x = self.network(batch)

                _, latent_distance = self.prototype_distances(x)
                similarities = self.distance_2_similarity(latent_distance)

                indicates = similarities.permute(2, 0, 1).flatten(1).argmax(dim=1)
                r_indicates = torch.stack([indicates // x.shape[1], indicates % x.shape[1]], -1).T

                max_similarities, max_indicates = similarities.max(dim=1)
                new_similarities, _ = max_similarities.max(dim=0)
                new_similarities = new_similarities.cpu().detach().view(-1,1)
                new_prototypes = x[r_indicates[0], r_indicates[1]].cpu().detach()

                selected_patch = (curr_similarities <= new_similarities).long()
                curr_prototypes = curr_prototypes*(1-selected_patch)+new_prototypes*selected_patch
                curr_similarities = curr_similarities*(1-selected_patch)+new_similarities*selected_patch

        for i, proto in enumerate(self.prototypes):
            start_idx, end_idx = 2 ** i - 1, 2 ** (i + 1) - 1
            proto.data.copy_(curr_prototypes[start_idx:end_idx].clone()).to(self.leaves)


                #self.prototype_vectors.data.copy_(x.permute(0,2,1)[r_indicates[0],r_indicates[1]].clone().detach()).to(data.y.device)
                #indicates = similarities.permute(2,0,1).flatten(1).argmax(dim=1)
                #torch.stack([indicates // 38, indicates % 21], -1)
                #r_indicates = torch.stack([indicates // x.shape[2], indicates % x.shape[2]], -1)

    def get_best_molecule(self, dl_proj):
        self.prototype_vectors = torch.cat(list(self.prototypes)).to(self.leaves)

        curr_ids = torch.zeros(self.prototype_shape[0])
        curr_similarities = torch.zeros(self.prototype_shape[0])

        for idx, batch in enumerate(dl_proj):
            batch = batch.to(self.leaves)

            with torch.no_grad():
                x = self.network(batch)

                _, latent_distance = self.prototype_distances(x)
                similarities = self.distance_2_similarity(latent_distance)

                indicates = similarities.permute(2, 0, 1).flatten(1).argmax(dim=1)
                r_indicates = torch.stack([indicates // x.shape[1], indicates % x.shape[1]], -1).T

                max_similarities, max_indicates = similarities.max(dim=1)
                new_similarities, _ = max_similarities.max(dim=0)
                new_similarities = new_similarities.cpu().detach().view(-1)


                selected_patch = (curr_similarities <= new_similarities).long()
                curr_similarities = curr_similarities*(1-selected_patch)+new_similarities*selected_patch
                curr_ids = curr_ids*(1-selected_patch)+(dl_proj.batch_size*idx+r_indicates[0].cpu())*selected_patch

        return curr_ids.long()

    def join(self):
        for param in self.features.parameters(): # We can't join (model is to big)
            param.requires_grad = False

        for param in self.add_on.parameters():
            param.requires_grad = True

        for param in self.prototypes:
            param.requires_grad = True

        self.leaves.requires_grad = True
