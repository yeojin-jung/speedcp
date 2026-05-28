import csv
import os
import time
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from speedcp.lambda_trace import lambda_path
from speedcp import SpeedCP
from speedcp.utils import *


# =========================
# Configurations
# =========================
BASE_SEED = 214
NREPEATS = 10
NCNT = 1000
NSAMPLE = 2000
NFEATURES = 1000
NMIXTURES = 3
TEST_PROP = 0.5
CALIB_PROP = 0.4
MIS_COVERAGE = 0.1

GAMMA_GRID = np.logspace(0, 2, 20)
MAX_STEPS = 200

DEFAULT_NOISE_LEVELS = (0.1, 0.1, 0.3)
DEFAULT_DIRICHLET_ALPHA = (2, 1, 1)

EXPERIMENT_1_NSAMPLES = [500, 2000, 5000]
EXPERIMENT_2_NOISE_LEVELS = [
    (0.05, 0.05, 0.05),
    (0.1, 0.1, 0.3),
    (0.5, 0.5, 1.0),
]
EXPERIMENT_3_DIRICHLET_ALPHAS = [
    (2, 1, 1),
    (1, 1, 1),
    (5, 1, 1),
]

OUTDIR = "results/synthetic/elbow_diagnostics"
os.makedirs(OUTDIR, exist_ok=True)


def _format_elbow_preview(arr: np.ndarray, max_items: int = 6) -> str:
    arr = np.asarray(arr)
    if arr.size == 0:
        return "[]"
    if arr.size <= max_items:
        return np.array2string(arr, precision=2, separator=", ")
    head = np.array2string(arr[:max_items], precision=2, separator=", ")
    return f"{head[:-1]}, ...]"


def print_run_header(experiment_name: str, condition_label: str, repeat_idx: int, seed: int):
    print(
        f"[{experiment_name}] {condition_label} | repeat {repeat_idx + 1}/{NREPEATS} | seed={seed}"
    )


def print_elbow_summary(result: Dict):
    lambda_preview = _format_elbow_preview(result["lambda_elbow_sizes"])
    s_path_preview = _format_elbow_preview(result["s_path_first_trace"])
    print(
        f"  selected gamma={result['gamma']:.4g}, lambda*={result['lambda_star']:.4g}, "
        f"n_cal={result['n_cal']}, n_test={result['n_test']}"
    )
    print(
        f"  lambda elbows: mean={result['lambda_elbow_mean']:.2f}, std={result['lambda_elbow_std']:.2f}, "
        f"mean/n={result['lambda_elbow_ratio_mean']:.3f}, preview={lambda_preview}"
    )
    print(
        f"  S-path elbows: mean={result['s_path_elbow_mean']:.2f}, std={result['s_path_elbow_std']:.2f}, "
        f"mean/n={result['s_path_elbow_ratio_mean']:.3f}, first-trace={s_path_preview}"
    )


def split_data(X, y, calib_prop=0.3, test_prop=0.1, random_state=127):
    n = len(X)
    n_tc = int(n * (1 - test_prop))
    test_idx = np.arange(n_tc, n)

    train_calib_idx = np.arange(n_tc)
    train_idx, calib_idx = train_test_split(
        train_calib_idx,
        test_size=calib_prop / (1 - test_prop),
        random_state=random_state,
    )
    data = {
        "train": (X[train_idx], y[train_idx], train_idx),
        "calib": (X[calib_idx], y[calib_idx], calib_idx),
        "test": (X[test_idx], y[test_idx], test_idx),
    }
    return data


def sample_MN(p, N):
    return np.random.multinomial(N, p, size=1)


def generate_W(n: int, K: int, dirichlet_alpha: Sequence[float]):
    W = np.zeros((n, K))
    probs = np.random.dirichlet(dirichlet_alpha, size=n)
    topics = np.random.choice(np.arange(K), n, replace=True)
    for k in range(K):
        inds = np.where(topics == k)[0]
        order = align_order(k, K)
        W[inds, :] = probs[np.ix_(inds, order)]

    anchor_ind = np.random.choice(np.arange(n), K, replace=False)
    W[anchor_ind, :] = np.eye(K)
    W = np.apply_along_axis(lambda x: x / np.sum(x), 1, W)
    return W


def generate_data(
    N: int,
    n: int,
    p: int,
    K: int,
    test_prop: float,
    dirichlet_alpha: Sequence[float] = DEFAULT_DIRICHLET_ALPHA,
    noise_levels: Tuple[float, float, float] = DEFAULT_NOISE_LEVELS,
):
    n_tc = int(n * (1 - test_prop))
    W_tc = generate_W(n_tc, K, dirichlet_alpha)

    W_test = np.random.dirichlet(dirichlet_alpha, size=n - n_tc)
    n_shuffle = int(0.3 * W_test.shape[0])
    shuffle_rows = np.random.choice(W_test.shape[0], size=n_shuffle, replace=False)
    for row in shuffle_rows:
        np.random.shuffle(W_test[row])
    W = np.vstack([W_tc, W_test])

    A = np.random.uniform(0, 1, size=(p, K))
    anchor_ind = np.random.choice(np.arange(p), K, replace=False)
    A[anchor_ind, :] = np.eye(K)
    A = np.apply_along_axis(lambda x: x / np.sum(x), 0, A)

    D0 = W @ A.T
    D = np.apply_along_axis(sample_MN, 1, D0, N).reshape(n, p)
    assert np.sum(D.sum(axis=1) != N) == 0

    X = D / N

    beta = np.random.uniform(1, 10, size=(K, 1))
    beta = beta / beta.sum()
    nonlin = W[:, 0] * beta[0] + W[:, 1] * beta[1] + W[:, 2] * beta[2]
    nonlin += np.sin(2 * np.pi * W[:, 0]) + W[:, 1] ** 2

    scale_1, scale_2, scale_3 = noise_levels
    topics = np.argmax(W, axis=1)
    noise_scale = np.where(topics == 0, scale_1, np.where(topics == 1, scale_2, scale_3))
    noise = np.random.normal(scale=noise_scale, size=n)
    y = nonlin.reshape(n, 1) + noise.reshape(n, 1)

    return X, y, D, W, A


def prepare_speedcp_inputs(
    seed: int,
    nsample: int,
    noise_levels: Tuple[float, float, float],
    dirichlet_alpha: Sequence[float],
):
    np.random.seed(seed)

    X, y, _, W_true, _ = generate_data(
        NCNT,
        nsample,
        NFEATURES,
        NMIXTURES,
        TEST_PROP,
        dirichlet_alpha=dirichlet_alpha,
        noise_levels=noise_levels,
    )
    splits = split_data(X, y, CALIB_PROP, TEST_PROP, seed)

    X_train, y_train, train_idx = splits["train"]
    X_calib, y_calib, calib_idx = splits["calib"]
    X_test, y_test, test_idx = splits["test"]

    reg = LinearRegression().fit(X_train, y_train.ravel())
    res_calib = np.abs(reg.predict(X_calib) - y_calib.ravel())

    W_hat, _ = run_plsi(X, NMIXTURES)
    P = get_component_mapping(W_true, W_hat)
    W_hat_aligned = W_hat @ P
    W_calib = W_hat_aligned[calib_idx, :]

    W_calib_clr = np.apply_along_axis(clr, 1, W_calib)
    W_calib_std = row_standardize(W_calib_clr)

    topic_calib = np.argmax(W_calib, axis=1)
    Phi_cal = np.eye(NMIXTURES)[topic_calib]
    Phi_cal[:, 0] = 1

    W_test = W_hat_aligned[test_idx, :]
    W_test_clr = np.apply_along_axis(clr, 1, W_test)
    W_test_std = row_standardize(W_test_clr)

    topic_test = np.argmax(W_test, axis=1)
    Phi_test = np.eye(NMIXTURES)[topic_test]
    Phi_test[:, 0] = 1

    res_test = np.abs(reg.predict(X_test) - y_test.ravel())

    return W_calib_std, Phi_cal, res_calib.ravel(), W_test_std, Phi_test, res_test.ravel()


def get_selected_gamma_elbow_trace(speedcp: SpeedCP, X_cal: np.ndarray, Phi_cal: np.ndarray, S_cal: np.ndarray):
    K = kernel(X_cal, X_cal, speedcp.gamma)
    res = lambda_path(
        S_cal.ravel(),
        Phi_cal,
        K,
        speedcp.alpha,
        max_steps=speedcp.max_steps,
        tol=speedcp.tol,
        thres=speedcp.thres,
        ridge=speedcp.ridge,
        verbose=False,
    )

    elbow_entry = speedcp.elbow_sizes_by_gamma.get(float(speedcp.gamma))
    if isinstance(elbow_entry, dict):
        elbow_sizes = np.asarray(elbow_entry["refit"], dtype=float)
    else:
        elbow_sizes = np.asarray(elbow_entry, dtype=float)

    lambdas = np.asarray(res["lambdas"], dtype=float)
    if elbow_sizes.shape[0] != lambdas.shape[0]:
        elbow_sizes = np.asarray([len(e) for e in res["Elbows"]], dtype=float)

    return lambdas, elbow_sizes, int(X_cal.shape[0])


def run_speedcp_condition(
    seed: int,
    nsample: int = NSAMPLE,
    noise_levels: Tuple[float, float, float] = DEFAULT_NOISE_LEVELS,
    dirichlet_alpha: Sequence[float] = DEFAULT_DIRICHLET_ALPHA,
):
    X_cal, Phi_cal, S_cal, X_test, Phi_test, S_test = prepare_speedcp_inputs(
        seed=seed,
        nsample=nsample,
        noise_levels=noise_levels,
        dirichlet_alpha=dirichlet_alpha,
    )

    start_time = time.time()
    speedcp_cv = SpeedCP(
        alpha=MIS_COVERAGE,
        max_steps=MAX_STEPS,
        eps=1e-3,
        tol=1e-6,
        thres=10.0,
        ridge=1e-8,
        start_side="left",
        gamma=None,
        gamma_grid=GAMMA_GRID,
        use_cv=True,
        randomize=True,
        verbose=False,
    )
    speedcp_cv.search_gamma_lambda_CV(X_cal, Phi_cal, S_cal, random_state=seed)
    lambdas, elbow_sizes, n_cal = get_selected_gamma_elbow_trace(speedcp_cv, X_cal, Phi_cal, S_cal)
    cutoffs_speedcp, _ = speedcp_cv.fit(X_cal, Phi_cal, S_cal, X_test, Phi_test, random_state=seed)
    covers_speedcp = (S_test <= cutoffs_speedcp).astype(int)
    time_speedcp = time.time() - start_time

    s_path_elbow_matrix = np.asarray(speedcp_cv.s_path_elbow_sizes, dtype=float)
    valid_s_path = s_path_elbow_matrix[~np.isnan(s_path_elbow_matrix)]
    first_valid_trace = np.array([], dtype=float)
    for row in s_path_elbow_matrix:
        valid = row[np.isfinite(row)]
        if valid.size:
            first_valid_trace = valid
            break

    return {
        "seed": int(seed),
        "nsample": int(nsample),
        "noise_levels": tuple(float(x) for x in noise_levels),
        "dirichlet_alpha": tuple(float(x) for x in dirichlet_alpha),
        "gamma": float(speedcp_cv.gamma),
        "lambda_star": float(speedcp_cv.lam),
        "speedcp_cutoffs": np.asarray(cutoffs_speedcp, dtype=float),
        "speedcp_covers": np.asarray(covers_speedcp, dtype=int),
        "speedcp_coverage": float(np.mean(covers_speedcp)),
        "time_speedcp": float(time_speedcp),
        "n_cal": int(n_cal),
        "n_test": int(X_test.shape[0]),
        "lambda_lambdas": lambdas,
        "lambda_elbow_sizes": elbow_sizes,
        "lambda_elbow_mean": float(np.mean(elbow_sizes)),
        "lambda_elbow_std": float(np.std(elbow_sizes)),
        "lambda_elbow_ratio_mean": float(np.mean(elbow_sizes / n_cal)),
        "lambda_elbow_ratio_std": float(np.std(elbow_sizes / n_cal)),
        "s_path_elbow_sizes": s_path_elbow_matrix,
        "s_path_first_trace": first_valid_trace,
        "s_path_elbow_mean": float(np.mean(valid_s_path)) if valid_s_path.size else np.nan,
        "s_path_elbow_std": float(np.std(valid_s_path)) if valid_s_path.size else np.nan,
        "s_path_elbow_ratio_mean": float(np.mean(valid_s_path / n_cal)) if valid_s_path.size else np.nan,
        "s_path_elbow_ratio_std": float(np.std(valid_s_path / n_cal)) if valid_s_path.size else np.nan,
    }


def save_trace(result: Dict, experiment_name: str, condition_label: str, repeat_idx: int):
    safe_label = condition_label.replace(" ", "_").replace("=", "-").replace(",", "_")
    path = os.path.join(OUTDIR, f"{experiment_name}_{safe_label}_repeat-{repeat_idx}.npz")
    np.savez_compressed(
        path,
        lambda_lambdas=np.asarray(result["lambda_lambdas"], dtype=float),
        lambda_elbow_sizes=np.asarray(result["lambda_elbow_sizes"], dtype=float),
        s_path_elbow_sizes=np.asarray(result["s_path_elbow_sizes"], dtype=float),
        speedcp_cutoffs=np.asarray(result["speedcp_cutoffs"], dtype=float),
        speedcp_covers=np.asarray(result["speedcp_covers"], dtype=np.int8),
        speedcp_coverage=np.float64(result["speedcp_coverage"]),
        time_speedcp=np.float64(result["time_speedcp"]),
        gamma=np.float64(result["gamma"]),
        lambda_star=np.float64(result["lambda_star"]),
        n_cal=np.int64(result["n_cal"]),
        n_test=np.int64(result["n_test"]),
        lambda_elbow_mean=np.float64(result["lambda_elbow_mean"]),
        lambda_elbow_std=np.float64(result["lambda_elbow_std"]),
        lambda_elbow_ratio_mean=np.float64(result["lambda_elbow_ratio_mean"]),
        lambda_elbow_ratio_std=np.float64(result["lambda_elbow_ratio_std"]),
        s_path_elbow_mean=np.float64(result["s_path_elbow_mean"]),
        s_path_elbow_std=np.float64(result["s_path_elbow_std"]),
        s_path_elbow_ratio_mean=np.float64(result["s_path_elbow_ratio_mean"]),
        s_path_elbow_ratio_std=np.float64(result["s_path_elbow_ratio_std"]),
    )
    return path


def plot_experiment(result_groups: List[List[Dict]], experiment_name: str, title: str, labels: List[str], path_type: str):
    fig, axes = plt.subplots(len(result_groups), 1, figsize=(10, 4 * len(result_groups)), sharex=False)
    if len(result_groups) == 1:
        axes = [axes]

    for ax, results, label in zip(axes, result_groups, labels):
        n_cal_vals = [result["n_cal"] for result in results]
        if path_type == "lambda":
            mean_vals = [result["lambda_elbow_mean"] for result in results]
            std_vals = [result["lambda_elbow_std"] for result in results]
            ratio_mean_vals = [result["lambda_elbow_ratio_mean"] for result in results]
        else:
            mean_vals = [result["s_path_elbow_mean"] for result in results]
            std_vals = [result["s_path_elbow_std"] for result in results]
            ratio_mean_vals = [result["s_path_elbow_ratio_mean"] for result in results]

        for repeat_idx, result in enumerate(results):
            if path_type == "lambda":
                x_vals = np.asarray(result["lambda_lambdas"], dtype=float)
                elbow_sizes = np.asarray(result["lambda_elbow_sizes"], dtype=float)
                ax.plot(
                    x_vals,
                    elbow_sizes,
                    marker="o",
                    linewidth=1.0,
                    markersize=2.5,
                    alpha=0.35,
                    color="tab:blue",
                    label="repeats" if repeat_idx == 0 else None,
                )
            else:
                s_path_matrix = np.asarray(result["s_path_elbow_sizes"], dtype=float)
                for row_idx, row in enumerate(s_path_matrix):
                    valid = np.isfinite(row)
                    if not np.any(valid):
                        continue
                    x_vals = np.arange(np.sum(valid))
                    ax.plot(
                        x_vals,
                        row[valid],
                        linewidth=0.8,
                        alpha=0.08,
                        color="tab:green",
                        label="test traces" if repeat_idx == 0 and row_idx == 0 else None,
                    )

        ax.axhline(
            float(np.mean(n_cal_vals)),
            linestyle="--",
            linewidth=1.2,
            color="tab:red",
            label=f"mean n={np.mean(n_cal_vals):.1f}",
        )
        ax.set_ylabel("|E|")
        ax.set_title(
            f"{label} | mean(|E|)={np.mean(mean_vals):.2f} +/- {np.std(mean_vals):.2f}, "
            f"mean(std(|E|))={np.mean(std_vals):.2f}, mean(|E|/n)={np.mean(ratio_mean_vals):.3f}"
        )
        if path_type == "lambda":
            ax.set_xscale("log")
            ax.set_xlabel("lambda")
        else:
            ax.set_xlabel("S-path step")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()

    path = os.path.join(OUTDIR, f"{experiment_name}_{path_type}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary_csv(rows: List[Dict]):
    path = os.path.join(OUTDIR, "elbow_summary.csv")
    fieldnames = [
        "experiment",
        "condition",
        "seed",
        "nsample",
        "noise_levels",
        "dirichlet_alpha",
        "gamma",
        "lambda_star",
        "speedcp_coverage",
        "time_speedcp",
        "n_cal",
        "n_test",
        "lambda_elbow_mean",
        "lambda_elbow_std",
        "lambda_elbow_ratio_mean",
        "lambda_elbow_ratio_std",
        "s_path_elbow_mean",
        "s_path_elbow_std",
        "s_path_elbow_ratio_mean",
        "s_path_elbow_ratio_std",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_experiment_1():
    result_groups = []
    labels = []
    rows = []

    for nsample in EXPERIMENT_1_NSAMPLES:
        label = f"NSAMPLE={nsample}"
        print(f"[Experiment 1] Starting condition: {label}")
        condition_results = []
        for repeat_idx in range(NREPEATS):
            seed = BASE_SEED + repeat_idx
            print_run_header("Experiment 1", label, repeat_idx, seed)
            result = run_speedcp_condition(
                seed=seed,
                nsample=nsample,
                noise_levels=DEFAULT_NOISE_LEVELS,
                dirichlet_alpha=DEFAULT_DIRICHLET_ALPHA,
            )
            print_elbow_summary(result)
            save_trace(result, "experiment1_nsample", label, repeat_idx)
            condition_results.append(result)
            rows.append({
                "experiment": "experiment1_nsample",
                "condition": label,
                "seed": result["seed"],
                "nsample": result["nsample"],
                "noise_levels": result["noise_levels"],
                "dirichlet_alpha": result["dirichlet_alpha"],
                "gamma": result["gamma"],
                "lambda_star": result["lambda_star"],
                "speedcp_coverage": result["speedcp_coverage"],
                "time_speedcp": result["time_speedcp"],
                "n_cal": result["n_cal"],
                "n_test": result["n_test"],
                "lambda_elbow_mean": result["lambda_elbow_mean"],
                "lambda_elbow_std": result["lambda_elbow_std"],
                "lambda_elbow_ratio_mean": result["lambda_elbow_ratio_mean"],
                "lambda_elbow_ratio_std": result["lambda_elbow_ratio_std"],
                "s_path_elbow_mean": result["s_path_elbow_mean"],
                "s_path_elbow_std": result["s_path_elbow_std"],
                "s_path_elbow_ratio_mean": result["s_path_elbow_ratio_mean"],
                "s_path_elbow_ratio_std": result["s_path_elbow_ratio_std"],
            })
        result_groups.append(condition_results)
        labels.append(label)
        print(f"[Experiment 1] Finished condition: {label}")

    plot_experiment(
        result_groups,
        "experiment1_nsample",
        "Experiment 1: Lambda-Path Elbow Size for Different NSAMPLE",
        labels,
        "lambda",
    )
    plot_experiment(
        result_groups,
        "experiment1_nsample",
        "Experiment 1: S-Path Elbow Size for Different NSAMPLE",
        labels,
        "s_path",
    )
    return rows


def run_experiment_2():
    result_groups = []
    labels = []
    rows = []

    for noise_levels in EXPERIMENT_2_NOISE_LEVELS:
        label = f"noise={noise_levels}"
        print(f"[Experiment 2] Starting condition: {label}")
        condition_results = []
        for repeat_idx in range(NREPEATS):
            seed = BASE_SEED + repeat_idx
            print_run_header("Experiment 2", label, repeat_idx, seed)
            result = run_speedcp_condition(
                seed=seed,
                nsample=NSAMPLE,
                noise_levels=noise_levels,
                dirichlet_alpha=DEFAULT_DIRICHLET_ALPHA,
            )
            print_elbow_summary(result)
            save_trace(result, "experiment2_noise", label, repeat_idx)
            condition_results.append(result)
            rows.append({
                "experiment": "experiment2_noise",
                "condition": label,
                "seed": result["seed"],
                "nsample": result["nsample"],
                "noise_levels": result["noise_levels"],
                "dirichlet_alpha": result["dirichlet_alpha"],
                "gamma": result["gamma"],
                "lambda_star": result["lambda_star"],
                "speedcp_coverage": result["speedcp_coverage"],
                "time_speedcp": result["time_speedcp"],
                "n_cal": result["n_cal"],
                "n_test": result["n_test"],
                "lambda_elbow_mean": result["lambda_elbow_mean"],
                "lambda_elbow_std": result["lambda_elbow_std"],
                "lambda_elbow_ratio_mean": result["lambda_elbow_ratio_mean"],
                "lambda_elbow_ratio_std": result["lambda_elbow_ratio_std"],
                "s_path_elbow_mean": result["s_path_elbow_mean"],
                "s_path_elbow_std": result["s_path_elbow_std"],
                "s_path_elbow_ratio_mean": result["s_path_elbow_ratio_mean"],
                "s_path_elbow_ratio_std": result["s_path_elbow_ratio_std"],
            })
        result_groups.append(condition_results)
        labels.append(label)
        print(f"[Experiment 2] Finished condition: {label}")

    plot_experiment(
        result_groups,
        "experiment2_noise",
        "Experiment 2: Lambda-Path Elbow Size for Different Noise Levels",
        labels,
        "lambda",
    )
    plot_experiment(
        result_groups,
        "experiment2_noise",
        "Experiment 2: S-Path Elbow Size for Different Noise Levels",
        labels,
        "s_path",
    )
    return rows


def run_experiment_3():
    result_groups = []
    labels = []
    rows = []

    for dirichlet_alpha in EXPERIMENT_3_DIRICHLET_ALPHAS:
        label = f"alpha={dirichlet_alpha}"
        print(f"[Experiment 3] Starting condition: {label}")
        condition_results = []
        for repeat_idx in range(NREPEATS):
            seed = BASE_SEED + repeat_idx
            print_run_header("Experiment 3", label, repeat_idx, seed)
            result = run_speedcp_condition(
                seed=seed,
                nsample=NSAMPLE,
                noise_levels=DEFAULT_NOISE_LEVELS,
                dirichlet_alpha=dirichlet_alpha,
            )
            print_elbow_summary(result)
            save_trace(result, "experiment3_dirichlet", label, repeat_idx)
            condition_results.append(result)
            rows.append({
                "experiment": "experiment3_dirichlet",
                "condition": label,
                "seed": result["seed"],
                "nsample": result["nsample"],
                "noise_levels": result["noise_levels"],
                "dirichlet_alpha": result["dirichlet_alpha"],
                "gamma": result["gamma"],
                "lambda_star": result["lambda_star"],
                "speedcp_coverage": result["speedcp_coverage"],
                "time_speedcp": result["time_speedcp"],
                "n_cal": result["n_cal"],
                "n_test": result["n_test"],
                "lambda_elbow_mean": result["lambda_elbow_mean"],
                "lambda_elbow_std": result["lambda_elbow_std"],
                "lambda_elbow_ratio_mean": result["lambda_elbow_ratio_mean"],
                "lambda_elbow_ratio_std": result["lambda_elbow_ratio_std"],
                "s_path_elbow_mean": result["s_path_elbow_mean"],
                "s_path_elbow_std": result["s_path_elbow_std"],
                "s_path_elbow_ratio_mean": result["s_path_elbow_ratio_mean"],
                "s_path_elbow_ratio_std": result["s_path_elbow_ratio_std"],
            })
        result_groups.append(condition_results)
        labels.append(label)
        print(f"[Experiment 3] Finished condition: {label}")

    plot_experiment(
        result_groups,
        "experiment3_dirichlet",
        "Experiment 3: Lambda-Path Elbow Size for Different Dirichlet Parameters",
        labels,
        "lambda",
    )
    plot_experiment(
        result_groups,
        "experiment3_dirichlet",
        "Experiment 3: S-Path Elbow Size for Different Dirichlet Parameters",
        labels,
        "s_path",
    )
    return rows


def main():
    print(f"Saving experiment outputs to {OUTDIR}")
    print(f"Running {NREPEATS} repeats per condition")
    rows = []
    rows.extend(run_experiment_1())
    rows.extend(run_experiment_2())
    rows.extend(run_experiment_3())
    summary_path = write_summary_csv(rows)
    print(f"Saved summary -> {summary_path}")
    print(f"Saved plots and traces -> {OUTDIR}")


if __name__ == "__main__":
    main()
