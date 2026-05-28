import os
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import QM7b
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool, global_max_pool

from sklearn.model_selection import train_test_split
from sklearn.manifold import MDS

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def select_target(data, TARGET_IDX):
    y = data.y.view(-1)
    data.y = y[TARGET_IDX:TARGET_IDX+1]
    return data

def add_node_feats(data):
    ei, ea = data.edge_index, data.edge_attr
    n = data.num_nodes
    x = torch.zeros((n, 1), dtype=torch.float)
    mask = (ei[0] == ei[1])
    if mask.sum() > 0:
        diag_src = ei[0, mask]
        x[diag_src] = ea[mask].view(-1, 1)
    else:
        deg = torch.bincount(ei[0], minlength=n).float().view(-1, 1)
        x = deg
    if ea.dim() == 1:
        data.edge_attr = ea.view(-1, 1)
    data.x = x
    return data


def scatter_mds(points2d, values, title, xlabel="PCA 1", ylabel="PCA 2", fname=None):
    plt.figure(figsize=(4.5, 4.0))
    sc = plt.scatter(points2d[:, 0], points2d[:, 1], c=values, s=28)
    cb = plt.colorbar(sc)
    cb.set_label("Value")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=160)
    plt.show()

def compute_coverage(y, up):
    return y <= up

def coverage_by_cluster(points2d, y, intervals_dict, k=4, seed=7, fname=None):
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    lab = km.fit_predict(points2d)

    clusters = np.arange(k)
    results = {name: np.zeros(k) for name in intervals_dict.keys()}
    sizes   = np.zeros(k, dtype=int)

    for c in clusters:
        idx = (lab == c)
        sizes[c] = idx.sum()
        for name, up in intervals_dict.items():
            cov = compute_coverage(y[idx], up[idx]).mean() if idx.any() else np.nan
            results[name][c] = cov

    # Plot bars, one figure per method (simplest)
    for name, covs in results.items():
        plt.figure(figsize=(7.2, 4.6))
        plt.bar(clusters, covs)
        plt.axhline(0.9, linestyle="--")  # target coverage line (adjust if you used different alpha)
        plt.title(f"Per-cluster coverage — {name}  (k={k})")
        plt.xlabel("Cluster index")
        plt.ylabel("Empirical coverage")
        for c in clusters:
            plt.text(c, covs[c] + 0.01, f"n={sizes[c]}", ha="center", va="bottom", fontsize=9)
        plt.ylim(0.0, 1.05)
        plt.tight_layout()
        if fname:
            base, ext = os.path.splitext(fname)
            plt.savefig(f"{base}_{name.replace(' ', '_')}.png", dpi=160)
        plt.show()

    return results, sizes

def get_scalar_target(batch):
    y = batch.y
    if y.dim() == 2:
        if y.size(1) == 1:
            y = y.view(-1)
        #elif y.size(1) == 14:
        #    y = y[:, TARGET_IDX]
        else:
            raise ValueError(f"Unexpected y shape: {y.shape}")
    else:
        y = y.view(-1)
    return y

def zscore(t, mean, std):
    return (t - mean) / (std if std > 0 else 1.0)

def unzscore(t, mean, std):
    return t * (std if std > 0 else 1.0) + mean

def train_one_epoch(model, opt, loader, device, y_mean, y_std, clip_grad=None):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        target = get_scalar_target(batch) 
        target = zscore(target, y_mean, y_std) 

        opt.zero_grad()
        pred, _ = model(batch)                            
        loss = F.smooth_l1_loss(pred, target)
        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        opt.step()

        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / max(n, 1)

@torch.no_grad()
def eval_loss(model, loader, device, y_mean, y_std):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        target = get_scalar_target(batch)
        target = zscore(target, y_mean, y_std)
        pred, _ = model(batch)
        total += F.mse_loss(pred, target).item() * batch.num_graphs
        n += batch.num_graphs
    return total / max(n, 1)

def fit_with_early_stopping(
    model, opt,
    train_loader_train, train_loader_val,
    device, y_mean, y_std,
    epochs=50, patience=10, clip_grad=None, scheduler=None
):
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, opt, train_loader_train, device, y_mean, y_std, clip_grad)
        val = eval_loss(model, train_loader_val, device, y_mean, y_std)
        print(f"Epoch {epoch:02d} | train={tr:.4f} | val(MSE)={val:.4f}")

        if scheduler is not None:
            try:
                scheduler.step(val)
            except TypeError:
                scheduler.step()

        if val + 1e-8 < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val

@torch.no_grad()
def collect_predictions(model, loader, device, y_mean, y_std):
    model.eval()
    yhats, ys, embs = [], [], []
    for batch in loader:
        batch = batch.to(device)
        pred, g = model(batch)                
        pred = unzscore(pred, y_mean, y_std)   
        y_true = get_scalar_target(batch)      

        yhats.append(pred.cpu().numpy())
        ys.append(y_true.cpu().numpy())
        embs.append(g.cpu().numpy())
    return np.concatenate(yhats), np.concatenate(ys), np.vstack(embs)
