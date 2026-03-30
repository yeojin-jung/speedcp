# In this script, we will test the effect of choosing different
# lambda / gamma values on coverage and prediction set size
import argparse
import numpy as np
import pandas as pd
import glob
import pickle
import os
import sys
import time

from joblib import Parallel, delayed

from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from FastKernCP.speedcp import SpeedCP
from FastKernCP.S_trace import S_path
from FastKernCP.lambda_trace import lambda_path
from FastKernCP.utils import (
    generate_data, split_data, run_plsi, get_component_mapping,
    clr, row_standardize, barycentric_to_cartesian, kernel
)

# =========================
# Configurations
# =========================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--nsample",
    type=int,
    default=1000,
    help="Total sample size NSAMPLE used in generate_data",
)
args = parser.parse_args()

NSAMPLE = args.nsample
print(f"[CONFIG] Using NSAMPLE = {NSAMPLE}", flush=True)

BASE_SEED = 214
NTRIALS = 50
NCNT = 1000
NSAMPLE = 1000
NFEATURES = 1000
NMIXTURES = 3
NBINS = 10

test_prop = 0.2
calib_prop = 0.4
alpha = 0.1

max_steps = 200
eps = 1e-03
tol = 1e-06
thres = 10.0
ridge = 1e-08
randomize = True

N_JOBS_GRID = 1
N_JOBS_SEED = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

OUTDIR = "mixture_outputs_hyper"
os.makedirs(OUTDIR, exist_ok=True)

def build_df_cover():
    """
    Run lambda_path over a gamma grid across NTRIALS seeds
    and construct df_cover: a reduced grid of (lambda, gamma)
    to be used later in the main experiment.
    """
    # grid of gammas you want to explore
    gamma_grid = np.logspace(0, 2, 20)

    results = []

    for i in range(NTRIALS):
        SEED = BASE_SEED + i
        np.random.seed(SEED)
        print(f"[build_df_cover] Trial {i}, SEED={SEED}", flush=True)

        # Generate data and splits
        X, y, D, W, A = generate_data(NCNT, NSAMPLE, NFEATURES, NMIXTURES, test_prop)
        splits = split_data(X, y, calib_prop, test_prop, SEED)

        X_train, y_train, train_idx = splits['train']
        X_calib, y_calib, calib_idx = splits['calib']
        X_test, y_test, test_idx = splits['test']

        # Train predictor
        reg = LinearRegression().fit(X_train, y_train.ravel())
        res_train = np.abs(reg.predict(X_train) - y_train.ravel())
        res_calib = np.abs(reg.predict(X_calib) - y_calib.ravel())
        res_test = np.abs(reg.predict(X_test) - y_test.ravel())

        # Estimate latent structures
        W_hat, A_hat = run_plsi(X, NMIXTURES)  # X: n x p frequency matrix
        P = get_component_mapping(W, W_hat)
        W_hat_aligned = W_hat @ P
        W_train = W_hat_aligned[train_idx, :]
        W_calib = W_hat_aligned[calib_idx, :]
        W_test = W_hat_aligned[test_idx, :]

        # Centered log ratio transform on W
        W_train_clr = np.apply_along_axis(clr, 1, W_train)
        W_calib_clr = np.apply_along_axis(clr, 1, W_calib)
        W_test_clr = np.apply_along_axis(clr, 1, W_test)

        # standardize
        W_train_ = row_standardize(W_train_clr)
        W_calib_ = row_standardize(W_calib_clr)
        W_test_ = row_standardize(W_test_clr)

        topic_calib = np.argmax(W_calib, axis=1)
        Phi_cal_bin = np.eye(NMIXTURES)[topic_calib]
        topic_test = np.argmax(W_test, axis=1)
        Phi_test_bin = np.eye(NMIXTURES)[topic_test]

        Phi_cal = Phi_cal_bin.copy()
        Phi_cal[:, 0] = 1  # intercept
        Phi_test = Phi_test_bin.copy()
        Phi_test[:, 0] = 1

        S_cal = np.asarray(res_calib, float).ravel()
        X_cal = np.asarray(X_calib, float)
        Phi_cal = np.asarray(Phi_cal, float)

        for g in gamma_grid:
            K = kernel(X_cal, X_cal, g)

            res = lambda_path(
                S_cal.ravel(), Phi_cal, K, alpha,
                max_steps=max_steps, tol=tol, thres=thres,
                ridge=ridge, verbose=False
            )
            lambdas = res['lambdas']

            for j, lam in enumerate(lambdas):
                results.append({
                    'trial': i,
                    'gamma': float(g),
                    'lambda': float(lam),
                    'v_est': res['v_arr'][j],
                    'eta_est': res['eta_arr'][j],
                })

    df_results = pd.DataFrame(results)

    cover_gamma = [gamma_grid[2 * i] for i in range(5)]
    df_cover = pd.DataFrame()

    for k, gam in enumerate(cover_gamma):
        selected = df_results[df_results['gamma'] == gam].copy()
        if selected.empty:
            continue

        selected = selected.sort_values(by='lambda', ascending=True).reset_index(drop=True)
        n = len(selected)
        if n == 0:
            continue
        idx = [0] + [int((n - 1) * i / 4) for i in range(1, 4)] + [n - 1]

        if k == 0 and n >= 4:
            idx[3] = int((n - 1) * 0.65)

        idx = sorted(set(max(0, min(n - 1, j)) for j in idx))

        selected_ = selected.iloc[idx]

        print("[build_df_cover] Selected (lambda, gamma) pairs:", flush=True)
        for _, row in selected_.iterrows():
            print(f"    trial={int(row['trial'])}, "
                f"gamma={row['gamma']:.4g}, "
                f"lambda={row['lambda']:.4g}", flush=True)
        df_cover = pd.concat([df_cover, selected_], ignore_index=True)

    print(f"[build_df_cover] df_cover shape = {df_cover.shape}", flush=True)
    return df_cover


def compute_grid_metrics(
    j,
    df_cover,
    X_cal, Phi_cal, S_cal,
    X_test, Phi_test, res_test, topics, bin_membership,
    alpha, eps, tol, ridge, randomize,
    SEED
):
    """Compute coverage metrics for a single (lambda, gamma) grid point."""
    lamb = df_cover.loc[j, "lambda"]
    gam = df_cover.loc[j, "gamma"]
    opt_v = df_cover.loc[j, "v_est"]
    opt_eta = df_cover.loc[j, "eta_est"]

    n_test = X_test.shape[0]
    out_S = np.empty(n_test, dtype=float)

    for i in range(n_test):
        alpha0 = np.random.uniform(-alpha, 1 - alpha) if randomize else alpha

        x_row = X_test[i].reshape(1, -1)     
        phi_row = Phi_test[i].reshape(1, -1)      
        X_all = np.vstack([X_cal, x_row])         
        Phi_all = np.vstack([Phi_cal, phi_row])   
        K_all = kernel(X_all, X_all, gam)

        res_S = S_path(
                S_cal, Phi_all, K_all, lamb, alpha,
                alpha0=alpha0, best_v=opt_v, best_eta=opt_eta,
                start_side="left", max_steps=100,
                eps=eps, tol=tol, ridge=ridge, verbose=False)
        out_S[i] = float(res_S["S_opt"])

    covers = (res_test <= out_S).astype(int)
    rows = []

    # Marginal coverage
    marginal_coverage = covers.mean()
    mean_S = out_S.mean()
    std_S = out_S.std()
    rows.append(
        {
            "seed": SEED,
            "lambda": lamb,
            "gamma": gam,
            "coverage": marginal_coverage,
            "shift_type": "Marginal",
            "cutoff": mean_S,
            "std_cutoff": std_S,
            "method": "grid",
        }
    )
    print(
        f"[Seed {SEED}] Grid j={j} | λ={lamb:.3e}, γ={gam:.3e} | "
        f"MarginalCov={marginal_coverage:.4f} | MeanCutoff={mean_S:.4f}",
        flush=True
    )

    # Topic-wise coverage
    for k in range(NMIXTURES):
        idx = np.where(topics == k)[0]
        if len(idx) == 0:
            continue
        cov_k = covers[idx].mean()
        mean_Sk = out_S[idx].mean()
        std_Sk = out_S[idx].std()
        rows.append(
            {
                "seed": SEED,
                "lambda": lamb,
                "gamma": gam,
                "coverage": cov_k,
                "shift_type": f"Cluster{k+1}",
                "cutoff": mean_Sk,
                "std_cutoff": std_Sk,
                "method": "grid",
            }
        )
        print(
            f"[Seed {SEED}]   Cluster{k+1} | Cov={cov_k:.4f} | "
            f"MeanCutoff={mean_Sk:.4f}",
            flush=True
        )

    # Bin-wise coverage based on KMeans bins
    for k in range(NBINS):
        idx = np.where(bin_membership == k)[0]
        if len(idx) == 0:
            continue
        cov_k = covers[idx].mean()
        mean_Sk = out_S[idx].mean()
        std_Sk = out_S[idx].std()
        rows.append(
            {
                "seed": SEED,
                "lambda": lamb,
                "gamma": gam,
                "coverage": cov_k,
                "shift_type": f"Bin{k+1}",
                "cutoff": mean_Sk,
                "std_cutoff": std_Sk,
                "method": "grid",
            }
        )
        print(
            f"[Seed {SEED}]   Bin{k+1} | Cov={cov_k:.4f} | "
            f"MeanCutoff={mean_Sk:.4f}",
            flush=True
        )
    return rows


def run_single_trial(SEED, df_cover):
    """Run one full trial (one seed) and return a DataFrame of results."""
    np.random.seed(SEED)
    print(f"Running seed {SEED}...")

    # Generate data and splits
    X, y, D, W, A = generate_data(NCNT, NSAMPLE, NFEATURES, NMIXTURES, test_prop)
    splits = split_data(X, y, calib_prop, test_prop, SEED)

    X_train, y_train, train_idx = splits["train"]
    X_calib, y_calib, calib_idx = splits["calib"]
    X_test, y_test, test_idx = splits["test"]

    # Train predictor and get residuals
    reg = LinearRegression().fit(X_train, y_train.ravel())
    res_train = np.abs(reg.predict(X_train) - y_train.ravel())
    res_calib = np.abs(reg.predict(X_calib) - y_calib.ravel())
    res_test = np.abs(reg.predict(X_test) - y_test.ravel())

    # Estimate latent structure via pLSI / topic model
    W_hat, A_hat = run_plsi(X, NMIXTURES)  # X: n x p frequency matrix
    P = get_component_mapping(W, W_hat)
    W_hat_aligned = W_hat @ P
    W_train = W_hat_aligned[train_idx, :]
    W_calib = W_hat_aligned[calib_idx, :]
    W_test = W_hat_aligned[test_idx, :]

    # CLR transform + standardize
    W_train_clr = np.apply_along_axis(clr, 1, W_train)
    W_calib_clr = np.apply_along_axis(clr, 1, W_calib)
    W_test_clr = np.apply_along_axis(clr, 1, W_test)

    W_train_ = row_standardize(W_train_clr)
    W_calib_ = row_standardize(W_calib_clr)
    W_test_ = row_standardize(W_test_clr)

    # Topic one-hot (simple Phi)
    topic_calib = np.argmax(W_calib, axis=1)
    Phi_cal_bin = np.eye(NMIXTURES)[topic_calib]
    topic_test = np.argmax(W_test, axis=1)
    Phi_test_bin = np.eye(NMIXTURES)[topic_test]

    # Add intercept in first column
    Phi_cal = Phi_cal_bin.copy()
    Phi_cal[:, 0] = 1.0
    Phi_test = Phi_test_bin.copy()
    Phi_test[:, 0] = 1.0

    S_cal = np.asarray(res_calib, float).ravel()
    X_cal_arr = np.asarray(X_calib, float)
    X_test_arr = np.asarray(X_test, float)
    Phi_cal_arr = np.asarray(Phi_cal, float)
    Phi_test_arr = np.asarray(Phi_test, float)

    # Precompute clustering bins for W_test using global kmeans
    W_cart = barycentric_to_cartesian(W_test)
    bin_membership = kmeans.predict(W_cart)
    topics = np.argmax(W_test, axis=1)

    # ======== 1. Fixed (lambda, gamma) grid, parallel over grid index ========
    #all_rows_nested = Parallel(n_jobs=N_JOBS_GRID)(
    #    delayed(compute_grid_metrics)(
    #        j,
    #        df_cover,
    #        X_cal_arr,
    #        Phi_cal_arr,
    #        S_cal,
    #        X_test_arr,
    #        Phi_test_arr,
    #        res_test,
    #        topics,
    #        bin_membership,
    #        alpha,
    #        eps,
    #        tol,
    #        ridge,
    #        randomize, 
    #        SEED,
    #    )
    #    for j in range(len(df_cover))
    #)
    all_rows_nested = [
        compute_grid_metrics(
            j,
            df_cover,
            X_cal_arr,
            Phi_cal_arr,
            S_cal,
            X_test_arr,
            Phi_test_arr,
            res_test,
            topics,
            bin_membership,
            alpha,
            eps,
            tol,
            ridge,
            randomize,
            SEED,
        )
        for j in range(len(df_cover))
    ]

    all_rows = [row for rows in all_rows_nested for row in rows]

    # ======== 2. SpeedCP with CV-selected hyperparameters (no split) ========
    print(f"Seed {SEED}: running SpeedCP with CV (no split)...")
    start_time = time.time()
    speedcp_cv = SpeedCP(
        alpha=alpha,
        max_steps=max_steps,
        eps=eps,
        tol=tol,
        thres=thres,
        ridge=ridge,
        start_side="left",
        gamma=None,
        gamma_grid=np.logspace(0, 2, 30),
        use_cv=True,
        use_split=False, 
        randomize=True,
        verbose=False,
    )

    cutoffs_cv, _ = speedcp_cv.fit(
        W_calib_, Phi_cal_arr, res_calib.ravel(), W_test_, Phi_test_arr, SEED
    )
    time_cv_total = time.time() - start_time
    time_cv_tune = getattr(speedcp_cv, "time_tune", np.nan)

    covers_cv = (res_test <= cutoffs_cv).astype(int)
    lam_cv = getattr(speedcp_cv, "lam", np.nan)
    gam_cv = getattr(speedcp_cv, "gamma", np.nan)

    # Marginal
    all_rows.append(
        {
            "seed": SEED,
            "lambda": lam_cv,
            "gamma": gam_cv,
            "coverage": covers_cv.mean(),
            "shift_type": "Marginal",
            "cutoff": cutoffs_cv.mean(),
            "std_cutoff": cutoffs_cv.std(),
            "method": "cv_full",
            "time_total": time_cv_total,
            "time_tune": time_cv_tune,
        }
    )
    print(
        f"[Seed {SEED}] CV(full): lambda={lam_cv:.3e}, gamma={gam_cv:.3e}, "
        f"MargCov={covers_cv.mean():.4f}, CutoffMean={cutoffs_cv.mean():.4f}, "
        f"time_total={time_cv_total:.2f}s, time_tune={time_cv_tune:.2f}s",
        flush=True,
    )

    # Topic-wise
    for k in range(NMIXTURES):
        idx = np.where(topics == k)[0]
        if len(idx) == 0:
            continue
        cov_k = covers_cv[idx].mean()
        mean_Sk = cutoffs_cv[idx].mean()
        std_Sk = cutoffs_cv[idx].std()
        all_rows.append(
            {
                "seed": SEED,
                "lambda": lam_cv,
                "gamma": gam_cv,
                "coverage": cov_k,
                "shift_type": f"Cluster{k+1}",
                "cutoff": mean_Sk,
                "std_cutoff": std_Sk,
                "method": "cv_full",
                "time_total": time_cv_total,
                "time_tune": time_cv_tune,
            }
        )

    # Bin-wise
    for k in range(NBINS):
        idx = np.where(bin_membership == k)[0]
        if len(idx) == 0:
            continue
        cov_k = covers_cv[idx].mean()
        mean_Sk = cutoffs_cv[idx].mean()
        std_Sk = cutoffs_cv[idx].std()
        all_rows.append(
            {
                "seed": SEED,
                "lambda": lam_cv,
                "gamma": gam_cv,
                "coverage": cov_k,
                "shift_type": f"Bin{k+1}",
                "cutoff": mean_Sk,
                "std_cutoff": std_Sk,
                "method": "cv_full",
                "time_total": time_cv_total,
                "time_tune": time_cv_tune,
            }
        )

    # ======== 3. SpeedCP with CV + split calibration ========
    print(f"Seed {SEED}: running SpeedCP with CV (split)...")
    start_time = time.time()
    speedcp_split_cv = SpeedCP(
        alpha=alpha,
        max_steps=max_steps,
        eps=eps,
        tol=tol,
        thres=thres,
        ridge=ridge,
        start_side="left",
        gamma=None,
        gamma_grid=np.logspace(0, 2, 30),
        use_cv=True,
        use_split=True,     # <-- half for tuning, half for calibration
        randomize=True,
        verbose=False,
    )

    cutoffs_split, _ = speedcp_split_cv.fit(
        W_calib_, Phi_cal_arr, res_calib.ravel(), W_test_, Phi_test_arr, SEED
    )
    time_split_total = time.time() - start_time
    time_split_tune = getattr(speedcp_split_cv, "time_tune", np.nan)

    covers_split = (res_test <= cutoffs_split).astype(int)
    lam_split = getattr(speedcp_split_cv, "lam", np.nan)
    gam_split = getattr(speedcp_split_cv, "gamma", np.nan)

    # Marginal
    all_rows.append(
        {
            "seed": SEED,
            "lambda": lam_split,
            "gamma": gam_split,
            "coverage": covers_split.mean(),
            "shift_type": "Marginal",
            "cutoff": cutoffs_split.mean(),
            "std_cutoff": cutoffs_split.std(),
            "method": "cv_split",
            "time_total": time_split_total,
            "time_tune": time_split_tune,
        }
    )
    print(
        f"[Seed {SEED}] CV(split): lambda={lam_split:.3e}, gamma={gam_split:.3e}, "
        f"MargCov={covers_split.mean():.4f}, CutoffMean={cutoffs_split.mean():.4f}, "
        f"time_total={time_split_total:.2f}s, time_tune={time_split_tune:.2f}s",
        flush=True,
    )

    # Topic-wise
    for k in range(NMIXTURES):
        idx = np.where(topics == k)[0]
        if len(idx) == 0:
            continue
        cov_k = covers_split[idx].mean()
        mean_Sk = cutoffs_split[idx].mean()
        std_Sk = cutoffs_split[idx].std()
        all_rows.append(
            {
                "seed": SEED,
                "lambda": lam_split,
                "gamma": gam_split,
                "coverage": cov_k,
                "shift_type": f"Cluster{k+1}",
                "cutoff": mean_Sk,
                "std_cutoff": std_Sk,
                "method": "cv_split",
                "time_total": time_split_total,
                "time_tune": time_split_tune,
            }
        )

    # Bin-wise
    for k in range(NBINS):
        idx = np.where(bin_membership == k)[0]
        if len(idx) == 0:
            continue
        cov_k = covers_split[idx].mean()
        mean_Sk = cutoffs_split[idx].mean()
        std_Sk = cutoffs_split[idx].std()
        all_rows.append(
            {
                "seed": SEED,
                "lambda": lam_split,
                "gamma": gam_split,
                "coverage": cov_k,
                "shift_type": f"Bin{k+1}",
                "cutoff": mean_Sk,
                "std_cutoff": std_Sk,
                "method": "cv_split",
                "time_total": time_split_total,
                "time_tune": time_split_tune,
            }
        )

    # ---- Save once at the end ----
    df_seed = pd.DataFrame(all_rows)
    save_path = os.path.join(OUTDIR, f"coverage_seed_{SEED}.pkl")
    df_seed.to_pickle(save_path)
    print(f"Seed {SEED}: saved -> {save_path}")

    return df_seed


def main_grid_parallel():
    all_dfs = []
    for t in range(NTRIALS):
        SEED = BASE_SEED + t
        try:
            df_seed = run_single_trial(SEED)
            all_dfs.append(df_seed)
        except Exception as e:
            print(f"Seed {SEED} failed with error: {e}")

    if len(all_dfs) > 0:
        df_all = pd.concat(all_dfs, ignore_index=True)
        save_path = os.path.join(OUTDIR, "mixture_hyper_outputs.pkl")
        df_all.to_pickle(save_path)
        print(f"All seeds combined and saved -> {save_path}")
    else:
        print("No successful runs.")

def main():
    df_cover = build_df_cover()
    seeds = [BASE_SEED + t for t in range(NTRIALS)]
    print(f"Running seeds: {seeds}")
    print(f"N_JOBS_SEED (parallel over seeds) = {N_JOBS_SEED}")
    
    dfs = Parallel(n_jobs=N_JOBS_SEED)(
        delayed(run_single_trial)(seed, df_cover) for seed in seeds
    )

    dfs = [df for df in dfs if df is not None]

    if len(dfs) > 0:
        df_all = pd.concat(dfs, ignore_index=True)
        save_path = os.path.join(OUTDIR, "mixture_hyper_outputs.pkl")
        df_all.to_pickle(save_path)
        print(f"All seeds combined and saved -> {save_path}")
    else:
        print("No successful runs.")


if __name__ == "__main__":
    test = []
    calib = []

    # for evaluating coverage on each bin
    for idx in range(50):
        SEED = BASE_SEED + idx
        np.random.seed(SEED)
        X, y, D, W, A = generate_data(NCNT, NSAMPLE, NFEATURES, NMIXTURES, test_prop)
        splits = split_data(X, y, calib_prop, test_prop, SEED)

        X_train, y_train, train_idx = splits['train']
        X_calib, y_calib, calib_idx = splits['calib']
        X_test, y_test, test_idx = splits['test']
        points = barycentric_to_cartesian(W[test_idx])
        test.append(points)
        points = barycentric_to_cartesian(W[calib_idx])
        calib.append(points)

    test = np.vstack(test)
    calib = np.vstack(calib)
    all_list = [calib, test]

    kde = gaussian_kde(calib.T, bw_method='scott')
    density = kde(calib.T)
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(calib, sample_weight=density)

    # save all_list
    with open(os.path.join(OUTDIR, 'all_list.pkl'), 'wb') as f:
        pickle.dump(all_list, f)

    main()