import numpy as np
import pandas as pd
from numpy.linalg import norm

from sklearn.metrics.pairwise import rbf_kernel

from scipy.sparse.linalg import svds
from scipy.optimize import linear_sum_assignment

import ternary
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cvxpy as cp
from tqdm import tqdm

def kernel(x, y, gamma):
    return rbf_kernel(x,y, gamma=gamma)

def pinball(beta0, y0, tau):
    """
    defines pinball loss
    """
    tmp = y0-beta0
    loss = np.sum(tau*tmp[tmp>0])+np.sum((tau-1)*tmp[tmp<=0])
    return loss

def clr(probs):
    continuous = np.log(probs + np.finfo(probs.dtype).eps)
    continuous -= continuous.mean(-1, keepdims=True)
    return continuous

def alr(probs):
    probs = probs.copy()
    probs /= probs[-1]
    #continuous = np.log(probs + np.finfo(probs.dtype).eps)
    return probs[:-1]

def clr_then_stack(data, K, p0):
    X1_clr = np.apply_along_axis(clr, 1, data[:,:K])
    X_clr = np.hstack([X1_clr,data[:,K:K+p0]])
    return X_clr

def row_standardize(matrix):
    row_means = np.mean(matrix, axis=1, keepdims=True)
    row_stds = np.std(matrix, axis=1, keepdims=True)
    row_stds[row_stds == 0] = 1.0
    standardized_matrix = (matrix - row_means) / row_stds
    return standardized_matrix

def run_plsi(X, k):
    U, L, V = svds(X, k)
    V  = V.T
    L = np.diag(L)
    J, H_hat = preconditioned_spa(U, k)

    W_hat = get_W_hat(U, H_hat)
    A_hat = get_A_hat(W_hat,X)      
    return W_hat, A_hat

def preprocess_U(U, K):
    for k in range(K):
        if U[0, k] < 0:
            U[:, k] = -1 * U[:, k]
    return U

def precondition_M(M, K):
    Q = cp.Variable((K, K), symmetric=True)
    objective = cp.Maximize(cp.log_det(Q))
    constraints = [cp.norm(Q @ M, axis=0) <= 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    Q_value = Q.value
    return Q_value

def preconditioned_spa(U, K, precondition=True):
    J = []
    M = preprocess_U(U, K).T
    if precondition:
        L = precondition_M(M, K)
        S = L @ M
    else:
        S = M
    
    for t in range(K):
            maxind = np.argmax(norm(S, axis=0))
            s = np.reshape(S[:, maxind], (K, 1))
            S1 = (np.eye(K) - np.dot(s, s.T) / norm(s) ** 2).dot(S)
            S = S1
            J.append(maxind)
    H_hat = U[J, :]
    return J, H_hat

def get_W_hat(U, H):
    projector = H.T.dot(np.linalg.inv(H.dot(H.T)))
    theta = U.dot(projector)
    theta_simplex_proj = np.array([_euclidean_proj_simplex(x) for x in theta])
    return theta_simplex_proj

def get_A_hat(W_hat, M):
    projector = (np.linalg.inv(W_hat.T.dot(W_hat))).dot(W_hat.T)
    theta = projector.dot(M)
    theta_simplex_proj = np.array([_euclidean_proj_simplex(x) for x in theta])
    return theta_simplex_proj

def _euclidean_proj_simplex(v, s=1):
    (n,) = v.shape
    if v.sum() == s and np.all(v >= 0):
        return v
    
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    
    theta = (cssv[rho] - s) / (rho + 1.0)
    w = (v - theta).clip(min=0)
    return w

def get_component_mapping(stats_1, stats_2):
    similarity = stats_1.T @ stats_2
    cost_matrix = -similarity
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    P = np.zeros_like(cost_matrix)
    P[row_ind, col_ind] = 1
    return P

def plot_ternary(data_points, cover_vector, title, ax):
    scale = 1  # Simplex sum should be 1
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)

    for i, (point, cover) in enumerate(zip(data_points, cover_vector)):
        color = 'red' if cover else 'blue'  # True = Red, False = Blue
        tax.scatter([point], marker='o', color=color, s=50)

    # Configure ternary plot
    tax.boundary(linewidth=1.5)  # Draw the simplex boundary
    tax.gridlines(multiple=0.2, color="gray", linestyle="dotted")  # Grid
    tax.left_axis_label("Component 1", fontsize=12)
    tax.right_axis_label("Component 2", fontsize=12)
    tax.bottom_axis_label("Component 3", fontsize=12)
    tax.ticks(axis='lbr', multiple=0.2, linewidth=1, offset=0.02)
    tax.clear_matplotlib_ticks()  # Remove extra ticks
    ax.set_title(title, fontsize=14)
    
def plot_ternary_size(data_points, cover_vector, title, ax,
                      vmin, vmax,
                      cmap = cm.plasma):
    scale = 1
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)
    scatter = tax.scatter(data_points, marker='o', c=cover_vector, 
                          cmap=cmap, s=50, vmin=vmin, vmax=vmax)

    tax.boundary(linewidth=1.5) 
    tax.gridlines(multiple=0.2, color="gray", linestyle="dotted")
    tax.left_axis_label("Component 1", fontsize=12)
    tax.right_axis_label("Component 2", fontsize=12)
    tax.bottom_axis_label("Component 3", fontsize=12)
    tax.ticks(axis='lbr', multiple=0.2, linewidth=1, offset=0.02)
    tax.clear_matplotlib_ticks()
    ax.set_title(title, fontsize=14)

    return scatter

def barycentric_to_cartesian(p):
    x = 0.5 * (2 * p[:, 1] + p[:, 2])
    y = (np.sqrt(3) / 2) * p[:, 2]
    return np.column_stack((x, y))

def aggregate_results(results):

    all_W = []
    all_covers_qkm = []
    all_covers_scp = []
    all_cutoffs_qkm = []
    all_cutoffs_scp = []
    all_time_qkm = []
    all_time_scp = []
    all_seeds = []

    for result in results:
        all_seeds.append(result["seed"])
        all_W.append(result["W_test"])
        all_covers_qkm.append(result["covers_qkm_rand"])
        all_covers_scp.append(result["covers_scp"])
        all_cutoffs_qkm.append(np.mean(result["cutoffs_qkm_rand"]))
        all_cutoffs_scp.append(result["cutoffs_scp"])
        all_time_qkm.append(result["time_qkm_rand"])
        all_time_scp.append(result["time_scp"])

    # Stack everything
    W_all = np.vstack(all_W)
    covers_qkm = np.hstack(all_covers_qkm)
    covers_scp = np.hstack(all_covers_scp)
    xy_qkm = barycentric_to_cartesian(W_all)

    # Per-seed summary
    summary_df = pd.DataFrame({
        "seed": all_seeds,
        "cutoff_qkm": all_cutoffs_qkm,
        "cutoff_scp": all_cutoffs_scp,
        "time_qkm": all_time_qkm,
        "time_scp": all_time_scp
    })

    return xy_qkm, covers_qkm, covers_scp, summary_df


def ternary_heatmap_all_methods(xy, cover_dict, method_names=None, cmap=plt.cm.RdYlBu, nbins=30):
    if method_names is None:
        method_names = list(cover_dict.keys())

    n_methods = len(method_names)
    n_cols = min(3, n_methods)
    n_rows = int(np.ceil(n_methods / n_cols))

    # Define triangle corners
    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    
    # Create triangular grid
    x_vals = np.linspace(0, 1, nbins)
    y_vals = np.linspace(0, np.sqrt(3)/2, nbins)
    Xg, Yg = np.meshgrid(x_vals, y_vals)
    grid_points = np.vstack([Xg.ravel(), Yg.ravel()]).T

    # Mask points outside the triangle
    A = triangle[0]
    B = triangle[1]
    C = triangle[2]
    def in_triangle(p):
        v0 = C - A
        v1 = B - A
        v2 = p - A
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1 - v - w
        return (u >= 0) & (v >= 0) & (w >= 0)

    mask = np.array([in_triangle(p) for p in grid_points])
    triangle_grid = grid_points[mask]

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=True)
    axs = np.array(axs).flatten()
    #norm = Normalize(vmin=0, vmax=1)
    norm = TwoSlopeNorm(vmin=0.7, vcenter=0.9, vmax=1.0)

    for i, method in enumerate(method_names):
        print(method)
        values = cover_dict[method]
        z_grid = np.full(len(triangle_grid), np.nan)

        # For each bin center, average values in a small neighborhood
        radius = 1.0 / nbins
        for j, center in enumerate(triangle_grid):
            dists = np.linalg.norm(xy - center, axis=1)
            neighbors = values[dists < radius]
            if len(neighbors) > 0:
                z_grid[j] = neighbors.mean()

        ax = axs[i]
        sc = ax.tripcolor(
            triangle_grid[:, 0], triangle_grid[:, 1], z_grid,
            shading='flat', cmap=cmap, norm=norm
        )

        ax.plot(*triangle[[0, 1]].T, color='black')
        ax.plot(*triangle[[1, 2]].T, color='black')
        ax.plot(*triangle[[2, 0]].T, color='black')
        ax.set_title(f"{method}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, np.sqrt(3)/2)
        ax.set_aspect('equal')

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    ax.set_xticks([])
    #ax.tick_params(axis='x', labelsize=12)
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Smoothed Coverage")
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout(rect=[0, 0, 0.9, 1])