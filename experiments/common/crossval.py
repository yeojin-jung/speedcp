"""Cross-validation utilities used by the CondConf baseline scripts."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from conditionalconformal import CondConf
from speedcp.utils import pinball


def runCV(
    X,
    scores,
    kernel,
    gamma,
    alpha,
    n_folds,
    min_radius,
    max_radius,
    num_radius,
    phi,
    seed=214,
):
    """Select the CondConf RKHS radius by K-fold pinball loss.

    Parameters follow the original experiment scripts. ``alpha`` is the target
    miscoverage level, so the fitted quantile is ``1 - alpha``.
    """
    X = np.asarray(X, dtype=float)
    scores = np.asarray(scores, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float)
    radii = np.linspace(float(min_radius), float(max_radius), int(num_radius))
    losses = np.zeros_like(radii, dtype=float)
    quantile = 1.0 - float(alpha)

    kfold = KFold(n_splits=int(n_folds), shuffle=True, random_state=seed)
    for fold_train, fold_val in kfold.split(X):
        X_train, X_val = X[fold_train], X[fold_val]
        scores_train, scores_val = scores[fold_train], scores[fold_val]
        phi_train = phi[fold_train]

        for idx, radius in enumerate(radii):
            params = {
                "kernel": kernel,
                "gamma": gamma,
                "lambda": 1.0 / float(radius),
            }
            model = CondConf(
                score_fn=lambda x, y: x[:, -1],
                Phi_fn=lambda x, p=phi_train.shape[1]: x[:, -p:],
                infinite_params=params,
            )
            model.setup_problem(
                np.hstack([X_train, phi_train]),
                np.zeros_like(scores_train),
                scores_train,
            )

            fold_cutoffs = []
            for x_val, phi_val in zip(X_val, phi[fold_val]):
                test_point = np.concatenate([x_val, phi_val]).reshape(1, -1)
                fold_cutoffs.append(
                    model.predict(
                        quantile,
                        test_point,
                        lambda cutoff, x: cutoff,
                        exact=False,
                    )
                )
            losses[idx] += pinball(np.asarray(fold_cutoffs), scores_val, quantile)

    return losses / int(n_folds), radii
