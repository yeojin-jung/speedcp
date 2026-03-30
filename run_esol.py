import time
import os
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

from scipy.linalg import orthogonal_procrustes

from moleculenet_helpers import *
from tinyGNN import TinyGNN

# required or runnning conditional-conformal
os.environ["MOSEK_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

from FastKernCP.speedcp import SpeedCP
from FastKernCP.utils import *

# download conditional-conformal (Gibbs et al., 2023)
# !git clone https://github.com/jjcherian/conditional-conformal.git
from conditionalconformal import CondConf
from experiments.crossval import runCV

# download PCP (Zhang et al., 2004)
# !git clone https://github.com/yaozhang24/pcp.git
from PCP.utils import PCP, RLCP


# =========================
# Configurations
# =========================
BASE_SEED  = 214
BATCH_SIZE = 32
EPOCHS     = 50
PATIENCE   = 10
HIDDEN     = 64
LR         = 1e-3
WD         = 1e-5

ROOT   = "data/MoleculeNet"
OUTDIR = "ESOL_outputs_all"
os.makedirs(OUTDIR, exist_ok=True)

DEVICE = torch.device("cpu")
REF_ART_PATH = os.path.join(OUTDIR, "reference_pca_scaler.joblib")
N_PCS = 3

def make_loader(dataset_small, idxs, shuffle=False, batch_size=BATCH_SIZE):
    return DataLoader([dataset_small[i] for i in idxs], batch_size=batch_size, shuffle=shuffle)

def load_or_fit_reference_pca(Z_train, random_state,
                              ref_art_path=REF_ART_PATH, n_components=N_PCS, scale=True):
    if os.path.exists(ref_art_path):
        art = joblib.load(ref_art_path)
        scaler_ref = art["scaler"] if scale else None
        pca_ref = art["pca"]
        return scaler_ref, pca_ref, False

    if scale:
        scaler_ref = StandardScaler().fit(Z_train)
        Z_train_ = scaler_ref.transform(Z_train)
    else:
        scaler_ref = None
        Z_train_ = Z_train

    pca_ref = PCA(n_components=n_components, random_state=random_state, whiten=False).fit(Z_train_)
    joblib.dump({"scaler": scaler_ref, "pca": pca_ref}, ref_art_path)
    return scaler_ref, pca_ref, True

def main(RUN):
    SEED = RUN + BASE_SEED
    set_seed(SEED)

    # =========================
    # Dataset & transforms
    # =========================
    name = "ESOL" 
    dataset = MoleculeNet(root=ROOT, name=name)

    # =========================
    # Splits: Train / Cal / Test (50/25/25 here)
    # =========================
    all_idx = np.arange(len(dataset))
    train_idx, hold_idx = train_test_split(all_idx, test_size=0.6,  random_state=SEED)   # 50% train
    cal_idx,   test_idx = train_test_split(hold_idx,  test_size=0.5, random_state=SEED)  # 25%/25%

    # Small val slice from TRAIN for early stopping
    train_ids, val_ids = train_test_split(train_idx, test_size=0.15, random_state=SEED)

    train_loader_train = make_loader(dataset, train_ids, shuffle=True)
    train_loader_val   = make_loader(dataset, val_ids,   shuffle=False)
    train_loader_full  = make_loader(dataset, train_idx, shuffle=False)  # predictions/embeddings
    cal_loader         = make_loader(dataset, cal_idx,   shuffle=False)
    test_loader        = make_loader(dataset, test_idx,  shuffle=False)

    # =========================
    # Target normalization
    # =========================
    y_train_vec = torch.tensor([dataset[i].y.item() for i in train_ids], dtype=torch.float)
    y_mean = y_train_vec.mean().item()
    y_std  = (y_train_vec.std(unbiased=False).item() if y_train_vec.numel() > 1 else 1.0)

    # ========= run training / caching =========
    model_outputs_path = f"esol_run_outputs_{SEED}.npz"
    out_full = os.path.join(OUTDIR, model_outputs_path)

    if os.path.exists(out_full):
        print(f"Model outputs already exist at {model_outputs_path}. Loading.")
        result = np.load(out_full)

        yhat_train = result["yhat_train"]
        yhat_cal = result["yhat_cal"]
        yhat_test = result["yhat_test"]

        y_train = result["y_train"]
        y_cal = result["y_cal"]
        y_test = result["y_test"]

        Z_train = result["Z_train"]
        Z_cal = result["Z_cal"]
        Z_test = result["Z_test"]

        res_train = np.abs(y_train - yhat_train)
        res_cal  = np.abs(y_cal  - yhat_cal)
        res_test = np.abs(y_test - yhat_test)
    else:
        in_dim   = dataset.num_node_features
        edge_dim = dataset.num_edge_features

        model = TinyGNN(
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden=HIDDEN,
            depth=2,
            dropout=0.1,
            pool="mean",
            use_bn=True,
            residual=True
        ).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

        best_val = fit_with_early_stopping(
            model, opt,
            train_loader_train, train_loader_val,
            DEVICE, y_mean, y_std,
            epochs=EPOCHS, patience=PATIENCE,
            clip_grad=1.0,
            scheduler=None
        )

        # ========= collect (train/cal/test) =========
        yhat_train, y_train, Z_train = collect_predictions(model, train_loader_full, DEVICE, y_mean, y_std)
        yhat_cal,   y_cal,   Z_cal   = collect_predictions(model, cal_loader,        DEVICE, y_mean, y_std)
        yhat_test,  y_test,  Z_test  = collect_predictions(model, test_loader,       DEVICE, y_mean, y_std)

        # Quick sanity:
        res_train = np.abs(y_train - yhat_train)
        res_cal   = np.abs(y_cal   - yhat_cal)
        res_test  = np.abs(y_test  - yhat_test)
        print(f"Cal MAE: {res_cal.mean():.4f} | Test MAE: {res_test.mean():.4f}")

        # ========= Save raw run outputs (pre-PCA) =========
        np.savez(
            out_full,
            y_mean=y_mean, y_std=y_std,
            y_train=y_train, yhat_train=yhat_train, Z_train=Z_train,
            y_cal=y_cal,     yhat_cal=yhat_cal,     Z_cal=Z_cal,
            y_test=y_test,   yhat_test=yhat_test,   Z_test=Z_test
        )
        print("Saved ->", out_full)
    
    # ========= Fixed StandardScaler + PCA across runs =========
    # Fit-once on TRAIN embeddings if reference artifacts not present; otherwise reuse.
    anchor_path = 'anchor_idx.npy'
    anchor_emb_path = 'Z_anchor_ref.npy'

    if os.path.exists(os.path.join(OUTDIR, anchor_path)):
        print("Loading existing anchor points for Procrustes alignment...")
        anchor_idx = np.load(os.path.join(OUTDIR, anchor_path))
        Z_anchor_ref = np.load(os.path.join(OUTDIR, anchor_emb_path))
        Z_anchor_run = Z_test[anchor_idx]

        # Align run -> reference
        R, _ = orthogonal_procrustes(Z_anchor_run, Z_anchor_ref) 
        Z_train_aligned = Z_train @ R
        Z_cal_aligned   = Z_cal   @ R
        Z_test_aligned  = Z_test  @ R

    else:
        # save anchor path
        print("Saving anchor points for Procrustes alignment...")
        anchor_idx = np.arange(Z_test.shape[0])
        np.save(os.path.join(OUTDIR, anchor_path), anchor_idx)
        np.save(os.path.join(OUTDIR, anchor_emb_path), Z_test[anchor_idx])
        Z_train_aligned = Z_train
        Z_cal_aligned   = Z_cal
        Z_test_aligned  = Z_test

    scaler_ref, pca_ref, fitted_now = load_or_fit_reference_pca(Z_train, random_state=SEED)
    if fitted_now:
        print(f"[REF] Fitted and saved scaler/PCA on this run's TRAIN embeddings -> {REF_ART_PATH}")
    else:
        print(f"[REF] Loaded existing scaler/PCA from -> {REF_ART_PATH}")

    # Transform CAL/TEST (and TRAIN if you want) with the SAME scaler/PCA
    if scaler_ref is not None:
        Z_train_std = scaler_ref.transform(Z_train_aligned)
        Z_cal_std   = scaler_ref.transform(Z_cal_aligned)
        Z_test_std  = scaler_ref.transform(Z_test_aligned)
    else:
        Z_train_std = Z_train_aligned
        Z_cal_std   = Z_cal_aligned
        Z_test_std  = Z_test_aligned

    Ztrain_pca = pca_ref.transform(Z_train_std)
    Zcal_pca   = pca_ref.transform(Z_cal_std)
    Ztest_pca  = pca_ref.transform(Z_test_std)

    alpha = 0.1

    # ========= CondConf (Gibbs et al., 2023) =========
    print("Starting CondConf...")
    k = 5
    gamma = 4
    minRad = 0.0001
    maxRad = 1
    numRad = 40

    start_time = time.time()
    X_calib_ = np.hstack([Zcal_pca, np.ones((Zcal_pca.shape[0],1))])
    X_test_ = np.hstack([Ztest_pca, np.ones((Ztest_pca.shape[0],1))])
    phiFn = lambda x : x[:, Zcal_pca.shape[1]:]
    phiCalib = phiFn(X_calib_)
    phiTest = phiFn(X_test_)

    allLosses, radii = runCV(Zcal_pca, res_cal, 'rbf', gamma, alpha, k,
                                        minRad, maxRad, numRad, phiCalib)
    selectedRadius = radii[np.argmin(allLosses)]
    infinite_params = {'kernel': 'rbf', 'gamma': gamma, 'lambda': 1 / selectedRadius}

    # return 
    scoreFn = lambda x, y: x[:, -1]  # absolute residuals already computed

    # Get cutoffs
    condCovProgram = CondConf(score_fn = scoreFn, 
                                Phi_fn = phiFn, 
                                infinite_params = infinite_params)
    condCovProgram.setup_problem(X_calib_, y_cal.ravel(), res_cal.ravel())
    cutoffs_cc = []
    i=0
    for x_val, y_val in zip(X_test_, res_test.ravel()):
        x = x_val.reshape(1,-1)
        cutoff = condCovProgram.predict(quantile=1-alpha,
                                        x_test=x,
                                        score_inv_fn = lambda s, x : [x - s, x + s],
                                        S_min=min(res_cal),
                                        S_max=max(res_cal),
                                        randomize=True,
                                        exact=False,
                                        threshold=1-alpha)
        cutoffs_cc.append(np.abs(cutoff))
        i+=1
    cutoffs_cc = np.array(cutoffs_cc)
    cutoffs_cc = np.concatenate(cutoffs_cc)
    covers_cc  = (res_test < cutoffs_cc).astype(int)
    time_cc = time.time()-start_time

    # ========= Split-CP baseline =========
    start_time = time.time()
    scoresCalib = res_cal
    nCalib = len(scoresCalib)
    cutoffs_scp = np.quantile(np.abs(scoresCalib), [(1 - alpha) * (1 + 1 / nCalib)])[0]
    covers_scp  = (np.abs(res_test) < cutoffs_scp).astype(int)
    time_scp = time.time()-start_time

    # ========= SpeedCP =========
    print("Starting SpeedCP...")
    start_time = time.time()
    speedcp = SpeedCP(
        alpha=alpha,
        max_steps=200,
        eps=1e-03,
        tol=1e-06,
        thres=10.0,
        ridge=1e-08,
        start_side='left',
        gamma=None,
        gamma_grid=np.logspace(-1, 1, 20),
        use_cv=True,
        randomize=True,
        verbose=False
    )

    Phi_cal  = np.ones((Zcal_pca.shape[0], 1))
    Phi_test = np.ones((Ztest_pca.shape[0], 1))

    cutoffs_speedcp, _ = speedcp.fit(Zcal_pca, Phi_cal, res_cal.ravel(),
                              Ztest_pca, Phi_test)
    covers_speedcp = (res_test <= cutoffs_speedcp).astype(int)
    print(f"Selected gamma: {speedcp.gamma:.4f}, lambda: {speedcp.lam:.4f}")
    speedcp_time = time.time()-start_time

    # === PCP ===
    start_time = time.time()
    R_train = res_train

    PCP_model = PCP()
    PCP_model.train(Z_train_std, R_train, info=True)
    cutoffs_pcp, covers_pcp = PCP_model.calibrate(Z_cal_std, res_cal, 
                                                  Z_test_std, res_test, alpha, finite=True)
    covers_pcp = np.array(covers_pcp)
    time_pcp = time.time()-start_time
    
    # === RLCP ===
    start_time = time.time()
    cutoffs_rlcp, covers_rlcp = RLCP(Ztrain_pca, Zcal_pca, res_cal, Ztest_pca, res_test, alpha, finite=True)
    covers_rlcp = np.array(covers_rlcp)
    time_rlcp = time.time()-start_time

    print(f"Cutoffs: SCP = {cutoffs_scp}, SpeedCP = {np.mean(cutoffs_speedcp)}, PCP = {np.mean(cutoffs_pcp)}, RLCP = {np.mean(cutoffs_rlcp)}, CondConf = {np.mean(cutoffs_cc)}")

    # ========= Save ALL results (all methods) =========
    save_path = os.path.join(OUTDIR, f"esol_fixed_pca_outputs_{SEED}.npz")
    np.savez_compressed(
        save_path,
        # --- metadata ---
        seed=np.int64(SEED),
        alpha=np.float64(alpha),

        # --- embeddings / residuals (for any post-hoc analysis) ---
        Ztrain_pca=Ztrain_pca, Zcal_pca=Zcal_pca, Ztest_pca=Ztest_pca,
        resid_train=res_train, resid_cal=res_cal, resid_test=res_test,

        # --- Split-CP ---
        scp_cutoffs=np.array(cutoffs_scp, dtype=float),     # scalar
        scp_covers=covers_scp.astype(np.int8),
        scp_time=np.float64(time_scp),

        # --- SpeedCP ---
        speedcp_cutoffs=np.asarray(cutoffs_speedcp, dtype=float),
        speedcp_covers=covers_speedcp.astype(np.int8),
        speedcp_time=np.float64(speedcp_time),
        speedcp_lambda=np.float64(speedcp.lam),
        speedcp_gamma=np.float64(speedcp.gamma),

        # --- PCP ---
        pcp_cutoffs=np.asarray(cutoffs_pcp, dtype=float),
        pcp_covers=covers_pcp.astype(np.int8),
        pcp_time=np.float64(time_pcp),

        # --- RLCP ---
        rlcp_cutoffs=np.asarray(cutoffs_rlcp, dtype=float),
        rlcp_covers=covers_rlcp.astype(np.int8),
        rlcp_time=np.float64(time_rlcp),

        # --- CondConf ---
        condconf_cutoffs=np.asarray(cutoffs_cc, dtype=float),
        condconf_covers=covers_cc.astype(np.int8),
        condconf_time=np.float64(time_cc),

        # --- store train/cal/test IDs for reproducibility ---
        train_idx=train_idx, cal_idx=cal_idx, test_idx=test_idx
    )
    print("Saved ->", save_path)

if __name__ == "__main__":
    for run in range(EPOCHS):
        main(run)