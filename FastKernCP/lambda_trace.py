import numpy as np
from FastKernCP.utils import kernel, pinball
from cvxopt import matrix, solvers

def _as_np_unique_sorted(a):
    a = np.asarray(a, dtype=int).ravel()
    if a.size == 0: return a
    return np.unique(a)

def lambda_path(S_vec, Phi, K, alpha, 
                max_steps=1000, eps = 1e-3, tol=1e-6, thres=10.0, ridge=1e-8, verbose=False):
    """
    Compute the full lambda path for the kernel quantile regression problem
    with pinball loss at level alpha.
    Parameters
    ----------
    S_vec : (n,) array-like
        The response values.
    Phi : (n,d) array-like or None
        The linear basis matrix. If None, only an intercept is used.
    K : (n,n) array-like
        The kernel matrix.
    alpha : float
        The quantile level in (0,1).
    max_steps : int, optional (default=1000)
        The maximum number of steps to take along the path.
    eps : float, optional (default=1e-3)
        The minimum relative change in lambda to continue the path.
    tol : float, optional (default=1e-6)
        A tolerance for numerical stability.
    thres : float, optional (default=10.0)
        A threshold for lambda to stop the path.
    ridge : float, optional (default=1e-8)
        A small ridge regularization to ensure numerical stability.
    verbose : bool, optional (default=False)
        If True, print progress information.
    Returns
    -------
    dict
        A dictionary containing:
        - "lambdas": array of lambda values along the path
        - "v_arr": array of dual variable v at each step
        - "eta_arr": array of linear coefficients eta at each step
        - "Elbows": list of elbow index sets at each step
        - "fit": array of fitted values at each step
        - "Csic": array of Csic values at each step
        - "steps": number of steps taken
    """
    S_vec = np.asarray(S_vec, dtype=float).ravel()
    n = S_vec.size
    K = np.asarray(K, dtype=float)

    if Phi is not None:
        Phi = np.asarray(Phi, dtype=float)
        d = Phi.shape[1]
    else:
        Phi = np.ones((n, 1), dtype=float)
        d = 1

    # ----- Init -----
    ini = lambda_init(S_vec, Phi, K, alpha)
    indE = _as_np_unique_sorted(ini["indE"])
    indL = _as_np_unique_sorted(ini["indL"])
    indR = _as_np_unique_sorted(ini["indR"])
    lam  = float(ini["lambda"])
    v    = np.asarray(ini["v"], dtype=float).copy()
    eta  = np.asarray(ini["eta"], dtype=float).copy() if d > 0 else float(ini["eta"])

    if verbose:
        print(f"[init] lambda={lam:.6g}; |E|={indE.size}, |L|={indL.size}, |R|={indR.size}")

    # storage
    eta_arr = np.zeros((max_steps+1, d))
    v_arr   = np.zeros((max_steps+1, n))
    Csic    = np.zeros(max_steps+1)
    fit     = np.zeros((max_steps+1, n))
    lambda_vals = np.zeros(max_steps+1)
    Elbows  = [None]*(max_steps+1)

    eta_arr[0] = eta
    v_arr[0] = v
    lambda_vals[0] = lam
    Elbows[0] = indE.copy()

    # current fit
    Kv = K @ v
    g_hat = Phi @ eta + Kv / lam

    csic = np.log(pinball(g_hat, S_vec, 1-alpha) / n) + (np.log(n)/(2*n)) * indE.size
    Csic[0] = csic
    fit[0] = g_hat

    k = 0
    while k < max_steps:
        k += 1
        if indL.size == 0 and indR.size == 0:
            if verbose: print("[stop] both L and R empty")
            break

        # ----- Step 1: Solve linear system for lambda step size -----
        E = indE.copy()
        m = E.size
        KEE = K[np.ix_(E, E)]

        PhiE = Phi[E, :]
        active_cols = np.where(np.any(PhiE != 0.0, axis=0))[0]
        PhiE_act = PhiE[:, active_cols]
        d_act = active_cols.size
        
        A_top = np.hstack((PhiE_act, KEE))
        A_bot = np.hstack((np.zeros((d_act, d_act)), PhiE_act.T))
        A = np.vstack((A_top, A_bot))
        b = np.concatenate((S_vec[E], np.zeros(d_act)))

        AtA = A.T @ A + ridge * np.eye(A.shape[1])
        Atb = A.T @ b
        delta = np.linalg.solve(AtA, Atb)

        b_eta     = np.zeros(d, dtype=float)
        b_eta_act = delta[:d_act]
        b_eta[active_cols] = b_eta_act
        b_v       = delta[d_act:]

        h_l = Phi @ b_eta + K[:, E] @ b_v  # (n, )

        # ----- Step 2: Find the next lambda and event -----
        cand_steps, cand_who, cand_dir = [], [], []

        # (a) point in E hits a bound: v_E -> -alpha or 1-alpha
        if m > 0:
            vE = v[E].astype(np.float64, copy=False)
            bv = np.asarray(b_v, dtype=np.float64, order='C')

            safe = np.isfinite(bv) & (np.abs(bv) > tol)
            step_to_lo = np.full(m, np.inf, dtype=np.float64)
            step_to_hi = np.full(m, np.inf, dtype=np.float64)

            step_to_lo[safe] = (-alpha        - vE[safe]) / bv[safe]
            step_to_hi[safe] = ((1.0 - alpha) - vE[safe]) / bv[safe]

            for loc in range(m):
                s1, s2 = step_to_lo[loc], step_to_hi[loc]
                s_candidates = [s for s in (s1, s2) if np.isfinite(s) and s < -tol]
                if not s_candidates:
                    continue
                t = max(s_candidates) 
                dir_flag = -1 if abs(t - s1) < abs(t - s2) else +1
                cand_steps.append(float(t))
                cand_who.append(('leave', int(E[loc])))
                cand_dir.append(dir_flag)

        # (b) residual zero for i in L∪R (indices are in notE by construction)
        LR = np.concatenate((indL, indR))
        denom = S_vec[LR] - h_l[LR]
        safe = np.abs(denom) > 1e-12
        if np.any(safe):
            ratio = np.full(LR.size, np.nan, dtype=float)
            ratio[safe] = (g_hat[LR][safe] - h_l[LR][safe]) / denom[safe]
            good = safe & np.isfinite(ratio) & (ratio <= 1.0)
            if np.any(good):
                steps = lam * (ratio[good] - 1.0)
                mask  = steps < -tol
                if np.any(mask):
                    for idx_i, t in zip(LR[good][mask], steps[mask]):
                        dir_flag = -1 if idx_i in indL else 1
                        cand_steps.append(float(t))
                        cand_who.append(('hit', int(idx_i)))
                        cand_dir.append(dir_flag)

        if not cand_steps:
            if verbose: print("[stop] no more candidate events")
            break

        cand_steps = np.asarray(cand_steps, float)
        imax = int(np.argmax(cand_steps)) 
        step = float(cand_steps[imax])
        ev_kind, ev_idx = cand_who[imax]
        ev_dir = cand_dir[imax]

        if verbose:
            print(f"[{k}] event={ev_kind} idx={ev_idx} step={step:.6g} dir={ev_dir}")

        # take step
        lam_next = lam + step
        if verbose:
            print(f"[{k}] lambda_next={lam_next:.6g}")

        # early stops
        if lam_next <= 0 or not np.isfinite(lam_next):
            if verbose: print("[stop] lambda hit nonpositive")
            break

        if np.abs(lam - lam_next) < eps and lam_next < thres:
            if verbose: print("[stop] descent too small")
            break
            
        # ----- Step 3: Update the next E, L, R -----
        if ev_kind == 'leave':
            indE = indE[indE != ev_idx]
            if ev_dir == -1:
                indL = _as_np_unique_sorted(np.append(indL, ev_idx))
            else:
                indR = _as_np_unique_sorted(np.append(indR, ev_idx))
        else:
            if ev_idx in indL:
                indL = indL[indL != ev_idx]
            elif ev_idx in indR:
                indR = indR[indR != ev_idx]
            indE = _as_np_unique_sorted(np.append(indE, ev_idx))

        if len(indE) == 0:
            if verbose: print("[stop] E is empty")
            break

        # ----- Step 4: Update v and eta -----
        v = np.zeros(n, dtype=float)
        v[indL] = -alpha
        v[indR] = 1 - alpha

        E = indE.copy()
        m = len(indE)

        KEE = K[np.ix_(E, E)]
        KEL, KER = K[np.ix_(E, indL)], K[np.ix_(E, indR)]
        PhiE = Phi[E, :]

        active_cols = np.where(np.any(PhiE != 0.0, axis=0))[0]
        PhiE_act = PhiE[:, active_cols]
        d_act = active_cols.size

        PhiL_act, PhiR_act = Phi[np.ix_(indL, active_cols)], Phi[np.ix_(indR, active_cols)]
        one_L = np.ones(len(indL))
        one_R = np.ones(len(indR))

        S_E  = np.asarray(S_vec[E],  float).ravel()
        S_E_eff = S_E - (-alpha * (KEL @ one_L) + (1 - alpha) * (KER @ one_R)) / lam_next
        b_bot = alpha * (PhiL_act.T @ one_L) - (1 - alpha) * (PhiR_act.T @ one_R)

        # Build the stacked linear operator H z ≈ b  (least-squares)
        # H = [[PhiE, KEE], [0, PhiE^T]], b = [S_E, 0]
        top = np.hstack([PhiE_act, KEE / lam_next])
        bot = np.hstack([np.zeros((d_act, d_act)), PhiE_act.T])
        H   = np.vstack([top, bot])
        b   = np.concatenate([S_E_eff, b_bot])
        P_np = H.T @ H + ridge * np.eye(d_act + m)
        q_np = -(H.T @ b)

        # Inequalities: box on v_E only
        #  v_E ≤ (1-α) → [0_dxm  I_m] z ≤ (1-α)·1
        # -v_E ≤ α     → [0_dxm -I_m] z ≤ α·1
        Zdm = np.zeros((m, d_act))
        G_np = np.vstack([
            np.hstack([Zdm,  np.eye(m)]),
            np.hstack([Zdm, -np.eye(m)])
        ])
        h_np = np.hstack([
            (1.0 - alpha) * np.ones(m),
            alpha * np.ones(m)
        ])

        # CVXOPT solve
        solvers.options['show_progress'] = False
        P = matrix(P_np, tc='d'); q = matrix(q_np, tc='d')
        G = matrix(G_np, tc='d'); h = matrix(h_np, tc='d')
        try:
            sol = solvers.qp(P, q, G, h)
            if sol['status'] != 'optimal':
                if verbose: print("Warning: QP status:", sol['status'])
                z = np.linalg.lstsq(H, b, rcond=1e-12)[0]
            else:
                z = np.array(sol['x']).flatten()
        except Exception as e:
            if verbose: print("QP exception:", repr(e))
            z = np.linalg.lstsq(H, b, rcond=1e-12)[0]

        eta = np.zeros(d, dtype=float)
        eta[active_cols] = z[:d_act]
        vE  = z[d_act:]
        v[E] = vE

        # update fit
        lam = lam_next
        Kv = K @ v
        g_hat = Phi @ eta + Kv / lam

        # store
        fit[k] = g_hat
        lambda_vals[k] = lam
        v_arr[k]    = v
        eta_arr[k]  = eta
        Elbows[k]   = E
        Csic[k]     = np.log(pinball(g_hat, S_vec, 1-alpha) / n) + (np.log(n)/(2*n)) * len(E)

    lambda_opt = lambda_vals[np.argmin(Csic[:k])]

    if verbose:
        print(f"[done] steps={k}, |E|={indE.size}, lambda_opt={lambda_opt:.6g}")

    return {
        "lambdas": lambda_vals[:k],
        "v_arr": v_arr[:k],
        "eta_arr": eta_arr[:k],
        "Elbows": Elbows[:k],
        "fit": fit[:k],
        "Csic": Csic[:k],
        "steps": k
    }


def lambda_init(S_vec, Phi, K, alpha, verbose=False, tol=1e-12):
    S_vec = np.asarray(S_vec, dtype=float).ravel()
    n = S_vec.size
    K = np.asarray(K, dtype=float)

    # Linear part (default to intercept if None)
    if Phi is None:
        Phi = np.ones((n, 1), dtype=float)
        d = 1
    else:
        Phi = np.asarray(Phi, dtype=float)
        d = int(Phi.shape[1])

    quant = np.quantile(S_vec, 1 - alpha, method="higher")
    order = np.argsort(np.abs(S_vec - quant))

    istar, k_star = None, None
    for i in order:
        k = int(np.argmax(np.abs(Phi[i, 1:]))) if d > 1 else 0 # choose non-intercept
        if np.abs(Phi[i, k]) > tol:
            istar, k_star = int(i), int(k)
            break

    if istar is None:
        raise ValueError("All rows of Phi appear to be (numerically) zero — cannot initialize λ.")

    S_star = float(S_vec[istar])
    notindE = np.setdiff1d(np.arange(n), np.array([istar]), assume_unique=True)

    eta_k_star = S_star / Phi[istar, k_star]
    g_hat = Phi[:, k_star] * eta_k_star

    indE = [istar]
    indR = np.where(S_vec > g_hat)[0].tolist()
    indL = np.where(S_vec < g_hat)[0].tolist()

    eps_bnd = 1e-4
    denom = Phi[istar, k_star]
    v_star = (alpha * Phi[notindE, k_star].sum() - Phi[indR, k_star].sum()) / denom
    v_star_clipped = float(np.clip(v_star, -alpha + eps_bnd, 1.0 - alpha - eps_bnd))
    if verbose:
        print(f"v_star: {v_star:.6g}, clipped: {v_star_clipped:.6g}")

    v_init = np.full(n, -alpha, dtype=float)
    v_init[istar] = v_star_clipped
    if len(indR) > 0:
        v_init[indR] = 1.0 - alpha

    # ----- compute initial lambda (step length along the chosen slice) -----
    f_hat = K @ v_init
    f_hat_star = float(g_hat[istar])

    ratio = Phi[notindE, k_star] / Phi[istar, k_star]
    denom_all = S_vec[notindE] - S_star * ratio
    num_all = f_hat[notindE] - f_hat_star * ratio

    safe = np.abs(denom_all) > tol
    lam = np.inf
    if np.any(safe):
        cands = num_all[safe] / denom_all[safe]
        pos = cands[(cands > 0) & np.isfinite(cands)]
        if pos.size > 0:
            lam = float(np.max(pos))

    # ----- initial eta -----
    eta_init = np.zeros(d, dtype=float)
    if np.isfinite(lam) and lam > tol:
        eta_init[k_star] = (S_star - (f_hat_star / lam)) / Phi[istar, k_star]

    if verbose:
        print(f"[lambda_init] istar={istar}, k_star={k_star}, lam={lam:.6g}, |E|={len(indE)} |L|={len(indL)} |R|={len(indR)}")

    return {
        "v": v_init,
        "eta": eta_init,
        "lambda": lam,
        "indE": indE,
        "indR": indR,
        "indL": indL
    }