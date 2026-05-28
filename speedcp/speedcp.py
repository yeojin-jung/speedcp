import time
import numpy as np
from speedcp.utils import *
from speedcp.lambda_trace import lambda_path
from speedcp.S_trace import S_path
from typing import Tuple
from sklearn.model_selection import KFold, train_test_split

class SpeedCP:
    """
    Fast Kernel-based Conformal Pipeline:
      1) search gamma over a grid and, for each gamma, compute a lambda-path
      2) pick the best (gamma, lambda) by validation error or CSIC
      3) for each test point, run `S_path` to get S_opt
    Parameters
    ----------
    alpha : float
        Quantile level (e.g., 0.1).
    max_steps : int
        Max steps for path solvers.
    eps : float
        Small S offset for S_path initializer.
    tol : float
        Numerical tolerance.
    thres : float
        Small-step threshold for lambda early stop in lambda_path.
    ridge : float
        Small l2 regularizer for QP subproblems.
    start_side : {'left','right'}
        Side to start S_path from.
    gamma : float or None
        If provided, use this gamma (skip grid search).
    gamma_grid : array-like
        Grid of gamma values to search if `gamma` is None.
    verbose : bool
        Verbose logging.
    """
    def __init__(
        self,
        alpha: float = 0.1,
        max_steps: int = 500,
        eps: float = 1e-3,
        tol: float = 1e-6,
        thres: float = 10.0,
        ridge: float = 1e-6,
        start_side: str = "left",
        gamma = None,
        gamma_grid: np.ndarray = np.logspace(0, 2, 50),
        use_cv = False,
        use_split = False,
        randomize = True,
        verbose = False,
    ):
        self.alpha = float(alpha)
        self.max_steps = int(max_steps)
        self.eps = float(eps)
        self.tol = float(tol)
        self.thres = float(thres)
        self.ridge = float(ridge)
        self.start_side = str(start_side)
        self.gamma = None if gamma is None else float(gamma)
        self.gamma_grid = np.asarray(gamma_grid, dtype=float)
        self.use_cv = bool(use_cv)
        self.use_split = bool(use_split)
        self.randomize = bool(randomize)
        self.verbose = bool(verbose)
        self.lam = None
        self.calib_v = None
        self.calib_eta = None
        self.sics = None
        self.best_idx = None
        self.elbow_sizes = None
        self.elbow_sizes_by_gamma = None
        self.s_path_elbow_sizes = None
        self.s_path_elbows_by_test = None

    @staticmethod
    def _as_np_unique_sorted(a):
        a = np.asarray(a, dtype=int).ravel()
        return np.unique(a) if a.size else a
    
    @staticmethod
    def _elbow_sizes_from_result(res):
        elbows = res.get("Elbows", [])
        return np.asarray([len(e) for e in elbows], dtype=int)

    def _store_path_state(self, res, idx: int):
        self.lam = float(res["lambdas"][idx])
        self.calib_v = res["v_arr"][idx, :].copy()
        self.calib_eta = res["eta_arr"][idx, :].copy() if res.get("eta_arr") is not None else None
        self.sics = np.asarray(res["Csic"], dtype=float).copy()
        self.best_idx = int(idx)
        self.elbow_sizes = self._elbow_sizes_from_result(res)

    def _refit_selected_lambda(self, X_cal: np.ndarray, Phi_cal: np.ndarray, S_cal: np.ndarray):
        """
        Recompute path coefficients on the calibration subset used by S_path,
        choosing the path point closest to the tuned lambda.
        """
        if self.gamma is None or self.lam is None:
            raise RuntimeError("Gamma and lambda must be set before refitting.")

        K = kernel(X_cal, X_cal, self.gamma)
        res = lambda_path(
            np.asarray(S_cal, float).ravel(), np.asarray(Phi_cal, float), K, self.alpha,
            max_steps=self.max_steps, tol=self.tol, thres=self.thres,
            ridge=self.ridge, verbose=self.verbose
        )
        idx = int(np.argmin(np.abs(res["lambdas"] - self.lam)))
        self._store_path_state(res, idx)
    
    # --------------------------------------------
    # Stage 1: search (or fix) gamma and best lambda
    # --------------------------------------------
    def search_gamma_lambda(self, X_cal: np.ndarray, Phi_cal: np.ndarray, S_cal: np.ndarray):
        """
        Grid search for gamma and lambda by CSIC,
        and stores: self.gamma, self.lam, self.calib_v, self.calib_eta, self.sics,
        self.elbow_sizes, self.elbow_sizes_by_gamma.
        """
        S_cal = np.asarray(S_cal, float).ravel()
        X_cal = np.asarray(X_cal, float)
        Phi_cal = np.asarray(Phi_cal, float)
        if self.gamma is not None:
            K = kernel(X_cal, X_cal, self.gamma)
            res = lambda_path(
                S_cal.ravel(), Phi_cal, K, self.alpha,
                max_steps=self.max_steps, tol=self.tol, thres=self.thres,
                ridge=self.ridge, verbose=self.verbose
            )
            best = int(np.argmin(res["Csic"]))
            self._store_path_state(res, best)
            self.elbow_sizes_by_gamma = {
                float(self.gamma): self.elbow_sizes.copy()
            }
            return

        best_sic = np.inf
        best_gamma = None
        best_v = None
        best_eta = None
        best_lambda = None
        best_idx = None
        elbow_sizes_by_gamma = {}
        best_elbow_sizes = None
        best_sics = None

        for g in self.gamma_grid:
            K = kernel(X_cal, X_cal, g)

            res = lambda_path(
                S_cal.ravel(), Phi_cal, K, self.alpha,
                max_steps=self.max_steps, tol=self.tol, thres=self.thres,
                ridge=self.ridge, verbose=False
            )
            elbow_sizes = self._elbow_sizes_from_result(res)
            elbow_sizes_by_gamma[float(g)] = elbow_sizes
            b = int(np.argmin(res["Csic"]))
            if res["Csic"][b] < best_sic:
                best_sic = float(res["Csic"][b])
                best_gamma = float(g)
                best_v = res["v_arr"][b, :].copy()
                best_eta = res["eta_arr"][b, :].copy() if res.get("eta_arr") is not None else None
                best_lambda = float(res["lambdas"][b])
                best_idx = b
                best_elbow_sizes = elbow_sizes.copy()
                best_sics = np.asarray(res["Csic"], dtype=float).copy()
            
        # store
        self.gamma = best_gamma
        self.sics = best_sics
        self.lam = best_lambda
        self.calib_v = best_v
        self.calib_eta = best_eta
        self.best_idx = best_idx
        self.elbow_sizes = best_elbow_sizes
        self.elbow_sizes_by_gamma = elbow_sizes_by_gamma
        
        if self.verbose:
            print(f"[gamma search] best γ={best_gamma:.6g}, λ*={best_lambda:.6g}, CSIC={best_sic:.6g}, Elbow size={np.mean(best_elbow_sizes)}")
    
    def search_gamma_lambda_CV(self, X, Phi, S, n_folds = 5,
                               random_state = 123):
        """
        Cross-validated gamma and lambda search
        and stores: self.gamma, self.lam, self.calib_v, self.calib_eta, self.sics,
        self.elbow_sizes, self.elbow_sizes_by_gamma.
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        opt_valid_err = [] # (n_gamma,)
        elbow_sizes_by_gamma = {}

        tau = 1 - self.alpha
        for gamma in self.gamma_grid:
            fold_err = []
            K_full = kernel(X, X, gamma)
            # --- CV loop ---
            for train_idx, valid_idx in kf.split(S):
                S_train, S_val = S[train_idx], S[valid_idx]
                Phi_train, Phi_val = Phi[train_idx, :], Phi[valid_idx, :]
                #X_train, X_val = X[train_idx, :], X[valid_idx, :]
                
                # fit path
                #K_train = kernel(X_train, X_train, gamma)
                K_train = K_full[np.ix_(train_idx, train_idx)]
                res = lambda_path(S_train.ravel(), Phi_train, K_train, self.alpha,
                            max_steps=self.max_steps, tol=self.tol, thres=self.thres,
                            ridge=self.ridge, verbose=False)
                elbow_sizes_by_gamma.setdefault(float(gamma), []).append(
                    self._elbow_sizes_from_result(res)
                )
                
                # validation fit
                #K_val = kernel(X_val, X_train, gamma)
                K_val = K_full[np.ix_(valid_idx, train_idx)]
                fit_val = Phi_val @ res['eta_arr'].T + (K_val @ res['v_arr'].T)/res['lambdas'][None, :]
                
                # quantile loss
                diff = S_val[:, None] - fit_val
                val_loss = np.where(diff > 0, tau * diff, (tau - 1) * diff)
                val_err = np.mean(val_loss, axis=0)

                # best lambda in this fold
                opt = np.argmin(val_err)
                fold_err.append(val_err[opt])

            # --- aggregate across folds ---
            mean_val_err = np.mean(fold_err)
            opt_valid_err.append(mean_val_err)
        
        # choose best gamma by CV error
        opt_idx = np.argmin(opt_valid_err)
        best_gamma = self.gamma_grid[opt_idx]

        K_full = kernel(X, X, best_gamma)
        res_full = lambda_path(S.ravel(), Phi, K_full, self.alpha,
                            max_steps=self.max_steps, tol=self.tol, thres=self.thres,
                            ridge=self.ridge, verbose=False)
        b = int(np.argmin(res_full["Csic"]))
        best_lambda = float(res_full["lambdas"][b])
        #best_v = res_full["v_arr"][b, :].copy()
        #best_eta = res_full["eta_arr"][b, :].copy() if res_full.get("eta_arr") is not None else None

        # store
        self.gamma = best_gamma
        self.valid_err = np.array(opt_valid_err, dtype=float)
        self._store_path_state(res_full, b)
        elbow_sizes_by_gamma[float(best_gamma)] = {
            "cv_folds": elbow_sizes_by_gamma.get(float(best_gamma), []),
            "refit": self.elbow_sizes.copy()
        }
        self.elbow_sizes_by_gamma = elbow_sizes_by_gamma
        if self.verbose:
            print(f"[gamma search] best γ={best_gamma:.6g}, λ*={best_lambda:.6g}, Validation Error={np.min(self.valid_err):.6g}")

    # -------------------------------------------------
    # Stage 2: compute S_opt for each test point (S_path)
    # -------------------------------------------------
    def fit(self, X_cal: np.ndarray, Phi_cal: np.ndarray, S_cal: np.ndarray,
            X_test: np.ndarray, Phi_test: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensure gamma/lambda are selected, then run S_path for each test point.
        Returns S_opt_array and S_init_array
        -------
        S_opt_array : (n_test,) array of optimal S for each test sample.
        S_init_array : (n_test,) array of initial S for each test sample.
        """
        # ensure we have gamma, lambda, v, eta
        start_time = time.time()
        if self.lam is None or self.gamma is None or self.calib_v is None:
            # if use_split, split calibration set to tune hyperparameters
            if self.use_split:
                calib_idx, valid_idx = train_test_split(
                    np.arange(len(X_cal)),
                    test_size = 0.5,
                    random_state = random_state)
                X_cal, X_val = X_cal[calib_idx, :], X_cal[valid_idx, :]
                Phi_cal, Phi_val = Phi_cal[calib_idx, :], Phi_cal[valid_idx, :]
                S_cal, S_val = S_cal[calib_idx], S_cal[valid_idx]
            else:
                X_val, Phi_val, S_val = X_cal, Phi_cal, S_cal
            if self.use_cv:
                self.search_gamma_lambda_CV(X_val, Phi_val, S_val)
            else:
                self.search_gamma_lambda(X_val, Phi_val, S_val)
            #if self.use_split:
            #    self._refit_selected_lambda(X_cal, Phi_cal, S_cal)
        self.time_tune = time.time()-start_time

        S_cal = np.asarray(S_cal, float).ravel()
        X_cal = np.asarray(X_cal, float)
        Phi_cal = np.asarray(Phi_cal, float)
        X_test = np.asarray(X_test, float)
        Phi_test = np.asarray(Phi_test, float)
        n_test = X_test.shape[0]
        out_S = np.empty(n_test, dtype=float)
        out_S_init = np.empty(n_test, dtype=float)
        self.s_path_elbows_by_test = []

        for i in range(n_test):
            u = np.random.uniform(-self.alpha, 1-self.alpha, size=1)[0]
            alpha0 = u if self.randomize else 1-self.alpha

            x_row = X_test[i].reshape(1, -1)          
            phi_row = Phi_test[i].reshape(1, -1)      
            X_all = np.vstack([X_cal, x_row])         
            Phi_all = np.vstack([Phi_cal, phi_row])   
            K_all = kernel(X_all, X_all, self.gamma)
            res_S = S_path(
                S_cal, Phi_all, K_all, self.lam, self.alpha,
                alpha0=alpha0, best_v=self.calib_v, best_eta=self.calib_eta,
                start_side=self.start_side, max_steps=self.max_steps,
                eps=self.eps, tol=self.tol, ridge=self.ridge, verbose=self.verbose
            )
            out_S[i] = float(res_S["S_opt"])
            out_S_init[i] = float(res_S["S_init"])
            self.s_path_elbows_by_test.append(self._elbow_sizes_from_result(res_S))

        if self.s_path_elbows_by_test:
            max_len = max(arr.size for arr in self.s_path_elbows_by_test)
            s_path_elbow_sizes = np.full((n_test, max_len), np.nan, dtype=float)
            for i, arr in enumerate(self.s_path_elbows_by_test):
                s_path_elbow_sizes[i, :arr.size] = arr
            self.s_path_elbow_sizes = s_path_elbow_sizes
        else:
            self.s_path_elbow_sizes = np.empty((0, 0), dtype=float)

        return out_S, out_S_init
    
    def predict(self, X_cal, Phi_cal, X_new, Phi_new) -> np.ndarray:
        """
        Given calibrated (gamma, lambda, v, eta), compute g_hat for new X_new, Phi_new.
        Note: requires that `search_gamma_lambda` has been run.
        Returns g_new_array
        -------
        g_new : (n_new,) model fit at lambda* using stored (v, eta).
        """
        if self.lam is None or self.gamma is None or self.calib_v is None:
            raise RuntimeError("Call `search_gamma_lambda` or `fit` first to set (gamma, lambda, v, eta).")
        X_cal = np.asarray(X_cal, float)
        Phi_cal = np.asarray(Phi_cal, float)
        X_new = np.asarray(X_new, float)
        Phi_new = np.asarray(Phi_new, float)
        K_new = kernel(X_new, X_cal, self.gamma)             
        Kv = K_new @ self.calib_v                            
        if self.calib_eta is None:
            eta_term = 0.0
        else:
            eta_term = Phi_new @ self.calib_eta              
        return eta_term + Kv / self.lam
