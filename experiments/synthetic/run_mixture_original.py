import time
import os
import numpy as np
import pickle
from scipy.stats import gaussian_kde

# required for running conditional-conformal
os.environ["MOSEK_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

from speedcp import SpeedCP
from speedcp.utils import *

# External baseline: install conditional-conformal separately.
from conditionalconformal import CondConf
from experiments.common.crossval import runCV

# External baseline: install PCP/RLCP separately.
from PCP.utils import PCP, RLCP

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


# =========================
# Configurations
# =========================
BASE_SEED = 214
NTRIALS = 50
NCNT = 1000
NSAMPLE = 2000
NFEATURES = 1000
NMIXTURES = 3
test_prop = 0.5
calib_prop = 0.4
alpha = 0.1

OUTDIR = "results/synthetic/mixture_original"
os.makedirs(OUTDIR, exist_ok=True)

def split_data(X, y, calib_prop=0.3, test_prop=0.1, random_state=127):
    n = len(X)
    n_tc = int(n*(1 - test_prop))
    test_idx = np.arange(n_tc, n)

    train_calib_idx = np.arange(n_tc)
    train_idx, calib_idx = train_test_split(
        train_calib_idx,
        test_size=calib_prop/(1-test_prop),
        random_state=random_state
    )
    data = {
        'train':   (X[train_idx],  y[train_idx],  train_idx),
        'calib':   (X[calib_idx],  y[calib_idx],  calib_idx),
        'test':    (X[test_idx],   y[test_idx],   test_idx)
    }
    return data

def sample_MN(p, N):
    return np.random.multinomial(N, p, size=1)

def barycentric_to_cartesian(p):
    """ Map (p1, p2, p3) probability vector to 2D Cartesian coordinates in simplex """
    x = 0.5 * (2*p[:,1] + p[:,2]) / (p[:,0] + p[:,1] + p[:,2])
    y = (np.sqrt(3)/2) * p[:,2] / (p[:,0] + p[:,1] + p[:,2])
    return np.column_stack((x, y))

def generate_W(n, K):
    alpha = [2] + [1]*(K-1)
    W = np.zeros((n,K))
    probs = np.random.dirichlet(alpha, size=n)
    topics = np.random.choice(np.arange(K),n,replace=True)
    for k in range(K):
        inds = np.where(topics==k)[0]
        order = align_order(k, K)
        W[inds,:] = probs[np.ix_(inds, order)]
        
    # generate pure doc
    anchor_ind = np.random.choice(np.arange(n), K, replace=False)
    W[anchor_ind, :] = np.eye(K)
    W = np.apply_along_axis(lambda x: x/np.sum(x), 1, W)
    return W

def generate_data(N,n,p,K,test_prop):
    n_tc = int(n*(1-test_prop))
    W_tc  = generate_W(n_tc, K)

    # generate test mixtures with covariate shift
    alpha = [2] + [1]*(K-1)
    W_test = np.random.dirichlet(alpha, size=n-n_tc)
    n_shuffle = int(0.3 * W_test.shape[0])
    shuffle_rows = np.random.choice(W_test.shape[0], size = n_shuffle, replace=False)
    for row in shuffle_rows:
        np.random.shuffle(W_test[row])
    W = np.vstack([W_tc, W_test])
   
    A = np.random.uniform(0,1,size=(p,K))
    anchor_ind = np.random.choice(np.arange(p), K, replace=False)
    A[anchor_ind, :] = np.eye(K)
    A = np.apply_along_axis(lambda x: x/np.sum(x), 0, A)

    D0 = W @ A.T
    D = np.apply_along_axis(sample_MN, 1, D0, N).reshape(n,p)
    assert np.sum(D.sum(axis=1)!=N)==0

    X = D/N

    n_covariate = W.shape[1]
    beta = np.random.uniform(1,10,size=(n_covariate,1))
    beta = beta/beta.sum()
    nonlin = (W[:,0]*beta[0]+W[:,1]*beta[1]+W[:,2]*beta[2])
    nonlin += np.sin(2*np.pi*W[:,0]) + W[:,1]**2

    scale_1 = 0.1
    scale_2 = 0.1
    scale_3 = 0.3
    topics = np.argmax(W, axis=1)
    noise = np.random.normal(scale=np.where(topics==1, scale_1,
                                            np.where(topics == 3, scale_2, scale_3)),
                                            size=n)
    y = nonlin.reshape(n,1) + noise.reshape(n,1)

    return X, y, D, W, A


def main():
    successful_runs = 0

    while successful_runs < NTRIALS:
        SEED = BASE_SEED + successful_runs
        np.random.seed(SEED)
        print(f"Attempt {SEED}: ", end="")

        X, y, D, W, A = generate_data(NCNT, NSAMPLE, NFEATURES, NMIXTURES, test_prop)
        splits = split_data(X, y, calib_prop, test_prop, SEED)

        X_train, y_train, train_idx = splits['train']
        X_calib, y_calib, calib_idx = splits['calib']
        X_test, y_test, test_idx = splits['test']

        # Train predictor
        reg = LinearRegression().fit(X_train, y_train.ravel())
        res_train = np.abs(reg.predict(X_train) - y_train.ravel())  
        res_calib = np.abs(reg.predict(X_calib) - y_calib.ravel())
        res_test =  np.abs(reg.predict(X_test) - y_test.ravel())

        # Estimate latent structures
        # Align estimated latent mixture propotions with true W
        # This step is necessary for measuring coverage consistently across bins/topics
        W_hat, A_hat = run_plsi(X, NMIXTURES) # X should be n x p frequency matrix
        P = get_component_mapping(W, W_hat)
        W_hat_aligned = W_hat @ P
        W_train = W_hat_aligned[train_idx, :]
        W_calib = W_hat_aligned[calib_idx, :]
        W_test = W_hat_aligned[test_idx, :]

        # Get l1 error of W_hat
        l1_err = np.sum(np.abs(W - W_hat_aligned))/W.shape[0]

        # Centered log ratio transform on W
        # log transform
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

        Phi_cal = Phi_cal_bin
        Phi_cal[:,0] = 1
        Phi_test = Phi_test_bin
        Phi_test[:,0] = 1

        # ========= Split-CP baseline =========
        print("Starting SplitCP...")
        start_time = time.time()
        nCalib = len(res_calib)
        cutoffs_scp = np.quantile(np.abs(res_calib), [(1 - alpha) * (1 + 1 / nCalib)])[0]
        covers_scp  = (np.abs(res_test) < cutoffs_scp).astype(int)
        print(f"Coverage: {np.mean(covers_scp)*100:.2f}%")
        time_scp = time.time()-start_time

        for k in range(NMIXTURES):
            idx = np.where(topic_test == k)[0]
            shiftcov = np.mean(covers_scp[idx])
            print(shiftcov)

        # =========== SpeedCP ============
        start_time = time.time()
        print("Starting SpeedCP...")
        speedcp_cv = SpeedCP(
            alpha=alpha,
            max_steps=200,
            eps=1e-03,
            tol=1e-06,
            thres=10.0,
            ridge=1e-08,
            start_side='left',
            gamma=None,
            gamma_grid=np.logspace(0, 2, 30),
            use_cv=True,
            randomize=True,
            verbose=False
        )
        cutoffs_speedcp, _ = speedcp_cv.fit(W_calib_, Phi_cal, res_calib.ravel(),
                                    W_test_, Phi_test)
        covers_speedcp = (res_test <= cutoffs_speedcp).astype(int)
        time_speedcp = time.time()-start_time
        print(f"Selected gamma: {speedcp_cv.gamma:.4f}, lambda: {speedcp_cv.lam:.4f}")
        print(f"Coverage: {np.mean(covers_speedcp)*100:.2f}%")


        # === PCP ===
        start_time = time.time()
        R_train = res_train

        PCP_model = PCP()
        PCP_model.train(W_train_, R_train, info=True)
        cutoffs_pcp, covers_pcp = PCP_model.calibrate(W_calib_, res_calib, 
                                                    W_test_, res_test, alpha, finite=True)
        covers_pcp = np.array(covers_pcp)
        time_pcp = time.time()-start_time

        for k in range(NMIXTURES):
            id = np.where(topic_test == k)[0]
            shiftcov = np.mean(covers_pcp[id])
            print(shiftcov)

        
        # === RLCP ===
        start_time = time.time()
        cutoffs_rlcp, covers_rlcp = RLCP(W_train_, W_calib_, res_calib, W_test_, res_test, alpha, finite=True)
        covers_rlcp = np.array(covers_rlcp)
        time_rlcp = time.time()-start_time

        for k in range(NMIXTURES):
            id = np.where(topic_test == k)[0]
            shiftcov = np.mean(covers_rlcp[id])
            print(shiftcov)


        # === CondConf ===
        print("Starting CondConf...")
        k = 5
        gamma = 4
        minRad = 0.0001
        maxRad = 1
        numRad = 40

        start_time = time.time()
        X_calib_ = np.hstack([W_calib_, Phi_cal])
        X_test_ = np.hstack([W_test_, Phi_test])
        phiFn = lambda x : x[:, W_calib_.shape[1]:]
        phiCalib = phiFn(X_calib_)
        phiTest = phiFn(X_test_)

        allLosses, radii = runCV(W_calib_, res_calib, 'rbf', gamma, alpha, k,
                                            minRad, maxRad, numRad, phiCalib)
        selectedRadius = radii[np.argmin(allLosses)]
        print(f"Selected lambda: {1/selectedRadius:.4f}")
        infinite_params = {'kernel': 'rbf', 'gamma': gamma, 'lambda': 1 / selectedRadius}

        scoreFn = lambda x, y: x[:, -1]  # absolute residuals already computed

        # Get cutoffs
        condCovProgram = CondConf(score_fn = scoreFn, 
                                    Phi_fn = phiFn, 
                                    infinite_params = infinite_params)
        condCovProgram.setup_problem(X_calib_, y_calib.ravel(), res_calib.ravel())
        cutoffs_cc = []
        i=0
        for x_val, y_val in zip(X_test_, res_test.ravel()):
            x = x_val.reshape(1,-1)
            print(f"Predicting {i+1}/{X_test_.shape[0]}", end='\r')
            cutoff = condCovProgram.predict(quantile=1-alpha,
                                            x_test=x,
                                            score_inv_fn = lambda s, x : [x - s, x + s],
                                            S_min=min(res_calib),
                                            S_max=max(res_calib),
                                            randomize=True,
                                            exact=False,
                                            threshold=1-alpha)
            cutoffs_cc.append(np.abs(cutoff))
            i+=1
        cutoffs_cc = np.array(cutoffs_cc)
        cutoffs_cc = np.concatenate(cutoffs_cc)
        covers_cc  = (res_test < cutoffs_cc).astype(int)
        time_cc = time.time()-start_time

        for k in range(NMIXTURES):
            id = np.where(topic_test == k)[0]
            shiftcov = np.mean(covers_cc[id])
            print(shiftcov)

        print(f"Cutoffs: SCP = {cutoffs_scp}, SpeedCP = {np.mean(cutoffs_speedcp)}, PCP = {np.mean(cutoffs_pcp)}, RLCP = {np.mean(cutoffs_rlcp)}, CondConf = {np.mean(cutoffs_cc)}")

        # ========= Save ALL results (all methods) ========
        save_path = os.path.join(OUTDIR, f"mixture_outputs_{SEED}.npz")
        np.savez_compressed(
            save_path,
            # --- metadata ---
            seed=np.int64(SEED),
            alpha=np.float64(alpha),
            W_err = np.float64(l1_err),

            # --- embeddings / residuals (for any post-hoc analysis) ---
            W_test=W_test,
            W_test_true = W[test_idx],
            resid_train=res_train, resid_cal=res_calib, resid_test=res_test,

            # --- SpeedCP ---
            speedcp_cutoffs=np.asarray(cutoffs_speedcp, dtype=float),
            speedcp_covers=covers_speedcp.astype(np.int8),
            speedcp_time=np.float64(time_speedcp),
            speedcp_lambda=np.float64(speedcp_cv.lam),
            speedcp_gamma=np.float64(speedcp_cv.gamma),

            # --- CondConf ---
            condconf_cutoffs=np.asarray(cutoffs_cc, dtype=float),
            condconf_covers=covers_cc.astype(np.int8),
            condconf_time=np.float64(time_cc),
            condconf_lambda=np.float64(1 / selectedRadius),

            # --- Split-CP ---
            scp_cutoffs=np.array(cutoffs_scp, dtype=float),     # scalar
            scp_covers=covers_scp.astype(np.int8),
            scp_time=np.float64(time_scp),

            # --- PCP ---
            pcp_cutoffs=np.asarray(cutoffs_pcp, dtype=float),
            pcp_covers=covers_pcp.astype(np.int8),
            pcp_time=np.float64(time_pcp),

            # --- RLCP ---
            rlcp_cutoffs=np.asarray(cutoffs_rlcp, dtype=float),
            rlcp_covers=covers_rlcp.astype(np.int8),
            rlcp_time=np.float64(time_rlcp),
        )
        print("Saved ->", save_path)
        successful_runs += 1
    

if __name__ == "__main__":
    main()
    test = []
    calib = []

    # for evaluating coverage on each bin
    for idx in range(NTRIALS):
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
