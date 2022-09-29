import torch
from sklearn.metrics import roc_auc_score
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def evaluate_clst(eval_dataloader, model, criterion, device):
    acc = []
    roc = []
    loss_list = []
    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = batch.to(device)
            ys_pred, _, _, _ = model(batch)

            loss = criterion(torch.log(ys_pred), batch.y.long().reshape(-1))

            _, prediction = torch.max(ys_pred, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y.reshape(-1)).cpu().numpy())

            y = batch.y.long().reshape(-1).cpu().numpy()
            pred = ys_pred.detach().cpu().numpy()

            roc.append(roc_auc_score(y, pred[:, 1]))

        eval_state = {'loss': np.average(loss_list),
                      'acc': np.concatenate(acc, axis=0).mean(),
                      'roc': np.array(roc).mean()}

    return eval_state


def evaluate_reg(eval_dataloader, model, criterion, device):
    result = []
    scores = []
    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = batch.to(device)
            y_pred, _, _, _ = model(batch)

            loss = criterion(y_pred, batch.y)

            y_pred = y_pred.cpu()
            y = batch.y.cpu()

            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            scores.append([mse, mae, r2])

        result.append(loss.item())

    scores = np.array(scores)
    return np.array(result).mean(), tuple(zip(scores.mean(axis=0), scores.std(axis=0)))
