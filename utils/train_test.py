import numpy as np
import torch
import math
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score


def _train_test(model, criterion, dataloader, optimizer, device, pt=0, cl=0, prob=0, ld=0, mse=0, task_type='cls', mean=0, std=1, kmeans=None):
    stats = {'pred': [], 'correct': []}
    parts = {'loss': [], 'penalty': [], 'cluster': [], 'probs': [], 'latent_distance': []}

    for batch in dataloader:
        batch = batch.to(device)

        if model.training:
            optimizer.zero_grad()

        ys_pred, min_distances, penalty, probs = model(batch)
        if task_type == "clst":
            y = batch.y
            if kmeans is not None:
                y = batch.y.view(-1, 1)
                y = torch.Tensor(kmeans.predict((y.cpu() - mean) / std)).long().to(model.leaves.device)

            loss = criterion(torch.log(ys_pred), y)
        else:
            leaves = model.get_leaves().squeeze().broadcast_to(probs.shape)
            y = (batch.y - mean) / std
            y = y.view(-1, 1)
            loss = torch.mean(torch.sum(probs * (y - leaves) ** 2, dim=1))

        cluster_cost, latent_distance_cost = model.get_costs(min_distances)
        max_probs, _ = torch.max(probs, dim=1)
        prob_cost = torch.mean(-torch.log(max_probs))

        if pt > 0:
            loss += pt * penalty.sum()
        if cl > 0:
            loss += cl * cluster_cost
        if prob > 0:
            loss + prob * prob_cost
        if ld > 0:
            loss += ld * latent_distance_cost

        if model.training:
            loss.backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
            optimizer.step()

        ys_pred = ys_pred.detach().cpu().numpy()
        y = y.detach().cpu()

        if task_type == "reg":
            ys_pred = (ys_pred * std) + mean

        stats['pred'].extend(ys_pred)
        stats['correct'].extend(y.detach().cpu())

        parts['loss'].append(loss.item())
        parts['penalty'].append(penalty.item())
        parts['cluster'].append(cluster_cost.item())
        parts['probs'].append(prob_cost.item())
        parts['latent_distance'].append(latent_distance_cost.item())

        if model.training:
            model.normalize()

    total_parts = {}
    total_stats = {}

    for key, values in parts.items():
        total_parts[key] = np.array(values).mean()

    if task_type == "clst":
        pred = np.array(stats['pred']).argmax(axis=1)
        total_stats['acc'] = accuracy_score(stats['correct'], pred)
    #        total_stats['roc'] = roc_auc_score(stats['correct'], np.array(stats['pred']),multi_class='ovo')
    else:
        try:
            total_stats['mse'] = mean_squared_error(stats['correct'], stats['pred'])
            total_stats['neg_spearman'] = -1 * spearmanr([x.item() for x in stats['correct']], [x.item() for x in stats['pred']])[0]

            total_stats['rmse'] = np.sqrt(mean_squared_error(stats['correct'], stats['pred']))
            total_stats['mae'] = mean_absolute_error(stats['correct'], stats['pred'])
            total_stats['r2'] = r2_score(stats['correct'], stats['pred'])
        except:
            pass

    return total_parts, total_stats


def train(model, criterion, dataloader, optimizer, device, pt=0, cl=0, prob=0, ld=0, mse=0, task_type='cls', mean=0, std=1, kmeans=None):
    model.train()
    return _train_test(model, criterion, dataloader, optimizer, device, pt, cl, prob, ld, mse, task_type, mean, std, kmeans)


def test(model, criterion, dataloader, optimizer, device, pt=0, cl=0, prob=0, ld=0, mse=0, task_type='cls', mean=0, std=1, kmeans=None):
    model.eval()
    with torch.no_grad():
        return _train_test(model, criterion, dataloader, optimizer, device, pt, cl, prob, ld, mse, task_type, mean, std, kmeans)
