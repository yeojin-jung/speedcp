import numpy as np
from FastKernCP.utils import *
from FastKernCP.lambda_trace import lambda_path
from FastKernCP.S_trace import S_path
from typing import Tuple

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
        randomize = True,
        verbose: bool = False,
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
        self.randomize = bool(randomize)
        self.verbose = bool(verbose)
        self.lam = None
        self.calib_v = None
        self.calib_eta = None
        self.sics = None
        self.best_idx = None

    @staticmethod
    def _as_np_unique_sorted(a):
        a = np.asarray(a, dtype=int).ravel()
        return np.unique(a) if a.size else a
    
    # --------------------------------------------
    # Stage 1: search (or fix) gamma and best lambda
    # --------------------------------------------
    def search_gamma_lambda(self, X_cal: np.ndarray, Phi_cal: np.ndarray, S_cal: np.ndarray):
        """
        Grid search for gamma and lambda by CSIC,
        and stores: self.gamma, self.lam, self.calib_v, self.calib_eta, self.sics.
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
            self.lam = float(res["lambdas"][best])
            self.calib_v = res["v_arr"][best, :].copy()
            self.calib_eta = res["eta_arr"][best, :].copy() if res.get("eta_arr") is not None else None
            self.sics = np.array([res["Csic"][best]], dtype=float)
            self.best_idx = best

        best_sic = np.inf
        best_gamma = None
        best_v = None
        best_eta = None
        best_lambda = None
        best_idx = None
        all_sics = []
        for g in self.gamma_grid:
            K = kernel(X_cal, X_cal, g)

            res = lambda_path(
                S_cal.ravel(), Phi_cal, K, self.alpha,
                max_steps=self.max_steps, tol=self.tol, thres=self.thres,
                ridge=self.ridge, verbose=False
            )
            b = int(np.argmin(res["Csic"]))
            all_sics.append(float(res["Csic"][b]))
            if res["Csic"][b] < best_sic:
                best_sic = float(res["Csic"][b])
                best_gamma = float(g)
                best_v = res["v_arr"][b, :].copy()
                best_eta = res["eta_arr"][b, :].copy() if res.get("eta_arr") is not None else None
                best_lambda = float(res["lambdas"][b])
                best_idx = b
        # store
        self.gamma = best_gamma
        self.sics = np.array(all_sics, dtype=float)
        self.lam = best_lambda
        self.calib_v = best_v
        self.calib_eta = best_eta
        self.best_idx = best_idx
        if self.verbose:
            print(f"[gamma search] best γ={best_gamma:.6g}, λ*={best_lambda:.6g}, CSIC={best_sic:.6g}")
    
    def search_gamma_lambda_CV(self, X, Phi, S, n_folds = 5,
                               random_state = 123):
        """
        Cross-validated gamma and lambda search
        and stores: self.gamma, self.lam, self.calib_v, self.calib_eta, self.sics.
        """
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        opt_lambdas = []   # (n_gamma,)
        opt_valid_err = [] # (n_gamma,)

        tau = 1 - self.alpha
        for gamma in self.gamma_grid:
            fold_val_err = []
            fold_lambdas = []
            fold_v = []
            fold_eta = []
            # --- CV loop ---
            for train_idx, valid_idx in kf.split(S):
                S_train, S_val = S[train_idx], S[valid_idx]
                Phi_train, Phi_val = Phi[train_idx, :], Phi[valid_idx, :]
                X_train, X_val = X[train_idx, :], X[valid_idx, :]
                
                # fit path
                K_train = kernel(X_train, X_train, gamma)
                res = lambda_path(S_train.ravel(), Phi_train, K_train, self.alpha,
                                max_steps=self.max_steps, tol=self.tol, verbose=False)
                
                # validation fit
                K_val = kernel(X_val, X_train, gamma)
                fit_val = Phi_val @ res['eta_arr'].T + (K_val @ res['v_arr'].T)/res['lambdas'][None, :]
                
                # quantile loss
                diff = S_val[:, None] - fit_val
                val_loss = np.where(diff > 0, tau * diff, (tau - 1) * diff)
                val_err = np.mean(val_loss, axis=0)

                # best lambda in this fold
                opt = np.argmin(val_err)
                fold_val_err.append(val_err[opt])
                fold_lambdas.append(res['lambdas'][opt])
                fold_v.append(res['v_arr'][opt, :])
                fold_eta.append(res['eta_arr'][opt, :])

            # --- aggregate across folds ---
            mean_val_err = np.mean(fold_val_err)
            opt_valid_err.append(mean_val_err)
        
        # choose best gamma by CV error
        opt_idx = np.argmin(opt_valid_err)
        best_gamma = self.gamma_grid[opt_idx]

        K_full = kernel(X, X, best_gamma)
        res_full = lambda_path(S.ravel(), Phi, K_full, self.alpha,
                            max_steps=self.max_steps, tol=self.tol,verbose=False)
        b = int(np.argmin(res_full["Csic"]))
        best_lambda = float(res_full["lambdas"][b])
        best_v = res_full["v_arr"][b, :].copy()
        best_eta = res_full["eta_arr"][b, :].copy() if res_full.get("eta_arr") is not None else None

        # store
        self.gamma = best_gamma
        self.valid_err = np.array(opt_valid_err, dtype=float)
        self.lam = best_lambda
        self.calib_v = best_v
        self.calib_eta = best_eta
        if self.verbose:
            print(f"[gamma search] best γ={best_gamma:.6g}, λ*={best_lambda:.6g}, Validation Error={np.min(self.valid_err):.6g}")

    # -------------------------------------------------
    # Stage 2: compute S_opt for each test point (S_path)
    # -------------------------------------------------
    def fit(self, X_cal: np.ndarray, Phi_cal: np.ndarray, S_cal: np.ndarray,
            X_test: np.ndarray, Phi_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensure gamma/lambda are selected, then run S_path for each test point.
        Returns S_opt_array and S_init_array
        -------
        S_opt_array : (n_test,) array of optimal S for each test sample.
        S_init_array : (n_test,) array of initial S for each test sample.
        """
        # ensure we have gamma, lambda, v, eta
        if self.lam is None or self.gamma is None or self.calib_v is None:
            if self.use_cv:
                self.search_gamma_lambda_CV(X_cal, Phi_cal, S_cal)
            else:
                self.search_gamma_lambda(X_cal, Phi_cal, S_cal)
        S_cal = np.asarray(S_cal, float).ravel()
        X_cal = np.asarray(X_cal, float)
        Phi_cal = np.asarray(Phi_cal, float)
        X_test = np.asarray(X_test, float)
        Phi_test = np.asarray(Phi_test, float)
        n_test = X_test.shape[0]
        out_S = np.empty(n_test, dtype=float)
        out_S_init = np.empty(n_test, dtype=float)

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
                start_side=self.start_side, max_steps=100,
                eps=self.eps, tol=self.tol, ridge=self.ridge, verbose=self.verbose
            )
            out_S[i] = float(res_S["S_opt"])
            out_S_init[i] = float(res_S["S_init"])
            
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