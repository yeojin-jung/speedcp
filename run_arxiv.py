import time
import os
import numpy as np

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

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# =========================
# Configurations
# =========================
ROOT   = "arxiv"
BASE_SEED = 214
NTOPICS = 5
NTRIALS = 20

def main():
    X = np.loadtxt('X_arxiv.csv', delimiter=',')
    y = np.loadtxt('y_arxiv.csv', delimiter=',')
    W = np.loadtxt('W_arxiv.csv', delimiter=',')
    print(f"W has shape {W.shape}")
    print(f"X has shape {X.shape}")
    print(f"y has shape {y.shape}")

    successful_runs = 0
    attempt = 0

    while successful_runs < NTRIALS:
        SEED = BASE_SEED + attempt
        np.random.seed(SEED)
        print(f"Attempt {SEED}: ", end="")

        try:
            sample_idx = np.random.choice(range(X.shape[0]), 2000, replace=False)
            X_sample = X[sample_idx]
            y_sample = y[sample_idx]
            W_sample = W[sample_idx]

            alpha = 0.1

            all_idx = np.arange(sample_idx)
            train_idx, hold_idx = train_test_split(all_idx, test_size=0.5,  random_state=BASE_SEED)  # 50% train
            cal_idx,   test_idx = train_test_split(hold_idx,  test_size=0.5, random_state=BASE_SEED)  # 25%/25%

            X_train, y_train, W_train = X_sample[train_idx], y_sample[train_idx], W_sample[train_idx]
            X_cal,   y_cal,   W_cal   = X_sample[cal_idx],   y_sample[cal_idx],   W_sample[cal_idx]
            X_test,  y_test,  W_test  = X_sample[test_idx],  y_sample[test_idx],  W_sample[test_idx]

            reg = LinearRegression().fit(X_train, y_train.ravel())
            res_train = np.abs(reg.predict(X_train) - y_train.ravel())
            res_cal = np.abs(reg.predict(X_cal) - y_cal.ravel())
            res_test =  np.abs(reg.predict(X_test) - y_test.ravel())

            topic = np.argmax(W, axis = 1)
            Phi_cal = np.eye(NTOPICS)[topic[cal_idx]]
            Phi_test = np.eye(NTOPICS)[topic[test_idx]]

            # ========= CondConf (Gibbs et al., 2023) =========
            print("Starting CondConf...")
            k = 5
            gamma = 4
            minRad = 0.0001
            maxRad = 1
            numRad = 40

            start_time = time.time()
            X_calib_ = np.hstack([W_cal, Phi_cal])
            X_test_ = np.hstack([W_test, Phi_test])
            phiFn = lambda x : x[:, W_cal.shape[1]:]
            phiCalib = phiFn(X_calib_)
            phiTest = phiFn(X_test_)

            allLosses, radii = runCV(W_cal, res_cal, 'rbf', gamma, alpha, k,
                                                minRad, maxRad, numRad, phiCalib)
            selectedRadius = radii[np.argmin(allLosses)]
            infinite_params = {'kernel': 'rbf', 'gamma': gamma, 'lambda': 1 / selectedRadius}

            # return 
            scoreFn = lambda x, y: x[:, -1] 
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
            start_time = time.time()
            print("Starting SpeedCP...")
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
            cutoffs_speedcp, _ = speedcp.fit(W_cal, Phi_cal, res_cal.ravel(),
                                                W_test, Phi_test)
            covers_speedcp = (res_test <= cutoffs_speedcp).astype(int)
            speedcp_time = time.time()-start_time

            # === PCP ===
            start_time = time.time()
            R_train = res_train

            PCP_model = PCP()
            PCP_model.train(X_train, R_train, info=True)
            cutoffs_pcp, covers_pcp = PCP_model.calibrate(X_cal, res_cal, 
                                                        X_test, res_test, alpha, finite=True)
            covers_pcp = np.array(covers_pcp)
            time_pcp = time.time()-start_time
            
            # === RLCP ===
            start_time = time.time()
            cutoffs_rlcp, covers_rlcp = RLCP(W_train, W_cal, res_cal, W_test, res_test, alpha, finite=True)
            covers_rlcp = np.array(covers_rlcp)
            time_rlcp = time.time()-start_time

            print(f"Cutoffs: SCP = {cutoffs_scp}, SpeedCP = {np.mean(cutoffs_speedcp)}, PCP = {np.mean(cutoffs_pcp)}, RLCP = {np.mean(cutoffs_rlcp)}, CondConf = {np.mean(cutoffs_cc)}")

            # ========= Save ALL results (all methods) =========
            save_path = os.path.join(ROOT, f"arxiv_outputs_{SEED}.npz")
            np.savez_compressed(
                save_path,
                # --- metadata ---
                seed=np.int64(SEED),
                alpha=np.float64(alpha),

                # --- embeddings / residuals (for any post-hoc analysis) ---
                W_test=W_test,
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
            )
            print("Saved ->", save_path)

        except Exception as e:
            print(f"Error skipped ({e}).")

        attempt += 1


if __name__ == "__main__":
    main()