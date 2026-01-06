import time
import os
import numpy as np
import pickle
import argparse
from scipy.stats import gaussian_kde

# required for running conditional-conformal (Gibbs et al., 2024)
os.environ["MOSEK_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

from speedcp.speedcp import SpeedCP
from speedcp.utils import *

# download conditional-conformal/conditionalconformal 
# from https://github.com/jjcherian/conditional-conformal.git
from conditionalconformal import CondConf
from experiments.crossval import runCV

# download PCP from https://github.com/yaozhang24/pcp.git
from PCP.utils import PCP, RLCP

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# =========================
# Default Configurations
# =========================
BASE_SEED = 214
DEFAULT_NTRIALS = 50
NSAMPLE = 2000
NCNT = 1000
NFEATURES = 1000   
NMIXTURES = 3

test_prop = 0.1
calib_prop = 0.3
alpha = 0.1

max_steps = 200
eps = 1e-03
tol = 1e-06
thres = 10.0
ridge = 1e-08
randomize = True

DEFAULT_OUTDIR = "../results/mixture"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictor",
        type=str,
        choices=["linreg", "rf", "nn"],
        default="linreg",
        help="Base regression model"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=NSAMPLE,
        help="Total sample size n"
    )
    parser.add_argument(
        "--ntrials",
        type=int,
        default=DEFAULT_NTRIALS,
        help="Number of trials to run in THIS job"
    )
    parser.add_argument(
        "--trial_offset",
        type=int,
        default=0,
        help="Global trial index offset (for parallelization across jobs)"
    )
    parser.add_argument(
        "--W_type",
        type=str,
        choices=["est", "true", "bad"],
        default="est",
        help="Which W to use for embedding: 'est' (estimated), 'true', or 'bad' (random Dirichlet)."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help="Output directory"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag to append to filenames"
    )

    return parser.parse_args()


def build_predictor(name, seed):
    if name == "linreg":
        return LinearRegression()
    elif name == "rf":
        return RandomForestRegressor(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1
        )
    elif name == "nn":
        return MLPRegressor(
            hidden_layer_sizes=(100, 100),
            activation="relu",
            max_iter=500,
            random_state=seed
        )
    else:
        raise ValueError(f"Unknown predictor: {name}")


def run_one_trial_speedcp(args, trial_idx):
    SEED = BASE_SEED + trial_idx
    np.random.seed(SEED)
    print(f"Trial (global) {trial_idx} [local  {trial_idx - args.trial_offset + 1}/{args.ntrials}] (SEED={SEED})")

    n = args.n
    X, y, D, W, A = generate_data(NCNT, n, NFEATURES, NMIXTURES, test_prop)
    print(f"Simulated {len(X)} data points.")
    splits = split_data(X, y, calib_prop, test_prop, SEED)

    X_train, y_train, train_idx = splits['train']
    X_calib, y_calib, calib_idx = splits['calib']
    X_test,  y_test,  test_idx  = splits['test']

    # -------------------------
    # 1) Train predictor
    # -------------------------
    reg = build_predictor(args.predictor, SEED)
    reg.fit(X_train, y_train.ravel())

    res_train = np.abs(reg.predict(X_train) - y_train.ravel())
    res_calib = np.abs(reg.predict(X_calib) - y_calib.ravel())
    res_test  = np.abs(reg.predict(X_test)  - y_test.ravel())

    # -------------------------
    # 2) Estimate / choose W embedding
    # -------------------------
    W_hat, A_hat = run_plsi(X, NMIXTURES)  # X: n x p frequency matrix
    P = get_component_mapping(W, W_hat)
    W_hat_aligned = W_hat @ P

    if args.W_type == "true":
        W_embed = W
        W_source = "true"

    elif args.W_type == "est":
        W_embed = W_hat_aligned
        W_source = "est"

    elif args.W_type == "bad":
        n, K = W.shape
        #W_embed = np.full((n, K), 1.0 / K, dtype=float)
        W_embed = np.random.dirichlet(alpha=np.ones(K), size=n)
        W_source = "bad"
    else:
        raise ValueError(f"Unknown W_type: {args.W_type}")
        
    l1_err = np.sum(np.abs(W - W_embed)) / W.shape[0]

    W_train = W_embed[train_idx, :]
    W_calib = W_embed[calib_idx, :]
    W_test  = W_embed[test_idx, :]

    # -------------------------
    # 3) CLR + standardization
    # -------------------------
    W_train_clr = np.apply_along_axis(clr, 1, W_train)
    W_calib_clr = np.apply_along_axis(clr, 1, W_calib)
    W_test_clr  = np.apply_along_axis(clr, 1, W_test)

    W_train_ = row_standardize(W_train_clr)
    W_calib_ = row_standardize(W_calib_clr)
    W_test_  = row_standardize(W_test_clr)

    topic_calib = np.argmax(W_calib, axis=1)
    Phi_cal_bin = np.eye(NMIXTURES)[topic_calib]

    topic_test = np.argmax(W_test, axis=1)
    Phi_test_bin = np.eye(NMIXTURES)[topic_test]

    Phi_cal = Phi_cal_bin.copy()
    Phi_cal[:, 0] = 1
    Phi_test = Phi_test_bin.copy()
    Phi_test[:, 0] = 1

    # -------------------------
    # 5) SpeedCP
    # -------------------------
    print("Starting SpeedCP...")
    start_time = time.time()
    speedcp_cv = SpeedCP(
        alpha=alpha,
        max_steps=max_steps,
        eps=eps,
        tol=tol,
        thres=thres,
        ridge=ridge,
        start_side='left',
        gamma=None,
        gamma_grid=np.logspace(0, 2, 30),
        use_cv=True,
        use_split=False, 
        randomize=True,
        verbose=False
    )
    cutoffs_speedcp, _ = speedcp_cv.fit(
        W_calib_, Phi_cal, res_calib.ravel(),
        W_test_,  Phi_test, SEED
    )
    covers_speedcp = (res_test <= cutoffs_speedcp).astype(int)
    time_speedcp = time.time() - start_time

    # -------------------------
    # 9) Save everything
    # -------------------------
    tag = f"pred-{args.predictor}_n{args.n}_W-{W_source}"
    if args.tag:
        tag += f"_{args.tag}"
    fname = f"mixture_{tag}_seed{SEED}.npz"
    save_path = os.path.join(args.outdir, fname)

    np.savez_compressed(
        save_path,
        # metadata
        seed=np.int64(SEED),
        alpha=np.float64(alpha),
        n=np.int64(n),
        predictor=str(args.predictor),
        W_source=str(W_source),
        W_err=np.float64(l1_err),

        # embeddings / residuals
        W_test=W_test,
        W_test_true=W[test_idx],
        resid_train=res_train,
        resid_cal=res_calib,
        resid_test=res_test,

        # SpeedCP
        speedcp_cutoffs=np.asarray(cutoffs_speedcp, dtype=float),
        speedcp_covers=covers_speedcp.astype(np.int8),
        speedcp_time=np.float64(time_speedcp),
        speedcp_lambda=np.float64(speedcp_cv.lam),
        speedcp_gamma=np.float64(speedcp_cv.gamma),
    )
    print("Saved ->", save_path)

def run_one_trial(args, trial_idx):
    SEED = BASE_SEED + trial_idx
    np.random.seed(SEED)
    print(f"Trial (global) {trial_idx} [local  {trial_idx - args.trial_offset + 1}/{args.ntrials}] (SEED={SEED})")

    n = args.n
    X, y, D, W, A = generate_data(NCNT, n, NFEATURES, NMIXTURES, test_prop)
    print(f"Simulated {len(X)} data points.")
    splits = split_data(X, y, calib_prop, test_prop, SEED)

    X_train, y_train, train_idx = splits['train']
    X_calib, y_calib, calib_idx = splits['calib']
    X_test,  y_test,  test_idx  = splits['test']

    # -------------------------
    # 1) Train predictor
    # -------------------------
    reg = build_predictor(args.predictor, SEED)
    reg.fit(X_train, y_train.ravel())

    res_train = np.abs(reg.predict(X_train) - y_train.ravel())
    res_calib = np.abs(reg.predict(X_calib) - y_calib.ravel())
    res_test  = np.abs(reg.predict(X_test)  - y_test.ravel())

    # -------------------------
    # 2) Estimate / choose W embedding
    # -------------------------
    W_hat, A_hat = run_plsi(X, NMIXTURES)  # X: n x p frequency matrix
    P = get_component_mapping(W, W_hat)
    W_hat_aligned = W_hat @ P

    if args.W_type == "true":
        W_embed = W
        W_source = "true"

    elif args.W_type == "est":
        W_embed = W_hat_aligned
        W_source = "est"

    elif args.W_type == "bad":
        n, K = W.shape
        #W_embed = np.full((n, K), 1.0 / K, dtype=float)
        W_embed = np.random.dirichlet(alpha=np.ones(K), size=n)
        W_source = "bad"
    else:
        raise ValueError(f"Unknown W_type: {args.W_type}")
        
    l1_err = np.sum(np.abs(W - W_embed)) / W.shape[0]

    W_train = W_embed[train_idx, :]
    W_calib = W_embed[calib_idx, :]
    W_test  = W_embed[test_idx, :]

    # -------------------------
    # 3) CLR + standardization
    # -------------------------
    W_train_clr = np.apply_along_axis(clr, 1, W_train)
    W_calib_clr = np.apply_along_axis(clr, 1, W_calib)
    W_test_clr  = np.apply_along_axis(clr, 1, W_test)

    W_train_ = row_standardize(W_train_clr)
    W_calib_ = row_standardize(W_calib_clr)
    W_test_  = row_standardize(W_test_clr)

    topic_calib = np.argmax(W_calib, axis=1)
    Phi_cal_bin = np.eye(NMIXTURES)[topic_calib]

    topic_test = np.argmax(W_test, axis=1)
    Phi_test_bin = np.eye(NMIXTURES)[topic_test]

    Phi_cal = Phi_cal_bin.copy()
    Phi_cal[:, 0] = 1
    Phi_test = Phi_test_bin.copy()
    Phi_test[:, 0] = 1

    # -------------------------
    # 4) Split CP baseline
    # -------------------------
    print("Starting SplitCP...")
    start_time = time.time()
    nCalib = len(res_calib)
    cutoffs_scp = np.quantile(
        np.abs(res_calib),
        [(1 - alpha) * (1 + 1 / nCalib)]
    )[0]
    covers_scp  = (np.abs(res_test) < cutoffs_scp).astype(int)
    time_scp = time.time() - start_time

    # -------------------------
    # 5) SpeedCP
    # -------------------------
    print("Starting SpeedCP...")
    start_time = time.time()
    speedcp_cv = SpeedCP(
        alpha=alpha,
        max_steps=max_steps,
        eps=eps,
        tol=tol,
        thres=thres,
        ridge=ridge,
        start_side='left',
        gamma=None,
        gamma_grid=np.logspace(0, 2, 30),
        use_cv=True,
        use_split=False, 
        randomize=True,
        verbose=False
    )
    cutoffs_speedcp, _ = speedcp_cv.fit(
        W_calib_, Phi_cal, res_calib.ravel(),
        W_test_,  Phi_test, SEED
    )
    covers_speedcp = (res_test <= cutoffs_speedcp).astype(int)
    time_speedcp = time.time() - start_time

    # -------------------------
    # 6) PCP
    # -------------------------
    print("Starting PCP...")
    start_time = time.time()
    R_train = res_train
    PCP_model = PCP()
    PCP_model.train(W_train_, R_train, info=True)
    cutoffs_pcp, covers_pcp = PCP_model.calibrate(
        W_calib_, res_calib,
        W_test_,  res_test,
        alpha, finite=True
    )
    covers_pcp = np.array(covers_pcp)
    time_pcp = time.time() - start_time

    # -------------------------
    # 7) RLCP
    # -------------------------
    print("Starting RLCP...")
    start_time = time.time()
    cutoffs_rlcp, covers_rlcp = RLCP(
        W_train_, W_calib_,
        res_calib, W_test_,
        res_test, alpha,
        finite=True
    )
    covers_rlcp = np.array(covers_rlcp)
    time_rlcp = time.time() - start_time

    # -------------------------
    # 8) CondConf
    # -------------------------
    print("Starting CondConf...")
    k = 5
    gamma = 4
    minRad = 0.0001
    maxRad = 1
    numRad = 40

    start_time = time.time()
    X_calib_ = np.hstack([W_calib_, Phi_cal])
    X_test_  = np.hstack([W_test_,  Phi_test])

    phiFn = lambda x: x[:, W_calib_.shape[1]:]
    phiCalib = phiFn(X_calib_)

    allLosses, radii = runCV(
        W_calib_, res_calib,
        'rbf', gamma, alpha, k,
        minRad, maxRad, numRad,
        phiCalib
    )
    selectedRadius = radii[np.argmin(allLosses)]
    infinite_params = {
        'kernel': 'rbf',
        'gamma': gamma,
        'lambda': 1 / selectedRadius
    }

    scoreFn = lambda x, y: x[:, -1]

    condCovProgram = CondConf(
        score_fn=scoreFn,
        Phi_fn=phiFn,
        infinite_params=infinite_params
    )
    condCovProgram.setup_problem(
        X_calib_,
        y_calib.ravel(),
        res_calib.ravel()
    )

    cutoffs_cc = []
    for i, (x_val, res_val) in enumerate(zip(X_test_, res_test.ravel())):
        x = x_val.reshape(1, -1)
        cutoff = condCovProgram.predict(
            quantile=1 - alpha,
            x_test=x,
            score_inv_fn=lambda s, x: [x - s, x + s],
            S_min=min(res_calib),
            S_max=max(res_calib),
            randomize=True,
            exact=False,
            threshold=1 - alpha
        )
        cutoffs_cc.append(np.abs(cutoff))
    cutoffs_cc = np.concatenate(cutoffs_cc)
    covers_cc  = (res_test < cutoffs_cc).astype(int)
    time_cc = time.time() - start_time

    # -------------------------
    # 9) Save everything
    # -------------------------
    tag = f"pred-{args.predictor}_n{args.n}_W-{W_source}"
    if args.tag:
        tag += f"_{args.tag}"
    fname = f"mixture_{tag}_seed{SEED}.npz"
    save_path = os.path.join(args.outdir, fname)

    print(f"Cutoffs: SCP = {cutoffs_scp}, SpeedCP = {np.mean(cutoffs_speedcp)}, PCP = {np.mean(cutoffs_pcp)}, RLCP = {np.mean(cutoffs_rlcp)}, CondConf = {np.mean(cutoffs_cc)}")

    np.savez_compressed(
        save_path,
        # metadata
        seed=np.int64(SEED),
        alpha=np.float64(alpha),
        n=np.int64(n),
        predictor=str(args.predictor),
        W_source=str(W_source),
        W_err=np.float64(l1_err),

        # embeddings / residuals
        W_test=W_test,
        W_test_true=W[test_idx],
        resid_train=res_train,
        resid_cal=res_calib,
        resid_test=res_test,

        # SpeedCP
        speedcp_cutoffs=np.asarray(cutoffs_speedcp, dtype=float),
        speedcp_covers=covers_speedcp.astype(np.int8),
        speedcp_time=np.float64(time_speedcp),
        speedcp_lambda=np.float64(speedcp_cv.lam),
        speedcp_gamma=np.float64(speedcp_cv.gamma),

        # CondConf
        condconf_cutoffs=np.asarray(cutoffs_cc, dtype=float),
        condconf_covers=covers_cc.astype(np.int8),
        condconf_time=np.float64(time_cc),
        condconf_lambda=np.float64(1 / selectedRadius),

        # Split-CP
        scp_cutoffs=np.array(cutoffs_scp, dtype=float),
        scp_covers=covers_scp.astype(np.int8),
        scp_time=np.float64(time_scp),

        # PCP
        pcp_cutoffs=np.asarray(cutoffs_pcp, dtype=float),
        pcp_covers=covers_pcp.astype(np.int8),
        pcp_time=np.float64(time_pcp),

        # RLCP
        rlcp_cutoffs=np.asarray(cutoffs_rlcp, dtype=float),
        rlcp_covers=covers_rlcp.astype(np.int8),
        rlcp_time=np.float64(time_rlcp),
    )
    print("Saved ->", save_path)

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ----- run trials -----
    for local_idx in range(args.ntrials):
        trial_idx = args.trial_offset + local_idx
        run_one_trial(args, trial_idx)

    # ----- barycentric / KDE block -----
    # Use the SAME global trial indices as above
    #test_points = []
    #calib_points = []

    #for local_idx in range(args.ntrials):
    #    trial_idx = args.trial_offset + local_idx
    #    SEED = BASE_SEED + trial_idx
    #    np.random.seed(SEED)

    #    # note: generate_data signature is (N, n, p, K, test_prop)
    #    X, y, D, W, A = generate_data(NCNT, args.n, NFEATURES, NMIXTURES, test_prop)
    #    splits = split_data(X, y, calib_prop, test_prop, SEED)

     #   _, _, train_idx = splits['train']
     #   _, _, calib_idx = splits['calib']
     #   X_test, y_test, test_idx = splits['test']

    #    pts_test  = barycentric_to_cartesian(W[test_idx])
    #    pts_calib = barycentric_to_cartesian(W[calib_idx])
    #    test_points.append(pts_test)
    #    calib_points.append(pts_calib)

    #test_points  = np.vstack(test_points)
    #calib_points = np.vstack(calib_points)
    #all_list = [calib_points, test_points]

    #with open(os.path.join(args.outdir, "all_list.pkl"), "wb") as f:
    #    pickle.dump(all_list, f)


if __name__ == "__main__":
    main()