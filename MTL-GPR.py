# -*- coding: utf-8 -*-
"""
Multi-Task Gaussian Process Regression (MT-GPR) for Alfalfa Quality Estimation

Paper: "Fusion of Physics-Informed Network and Three-Dimensional Allometric Operator
       for UAV-Based Multi-Temporal Estimation of Alfalfa Quality"
Journal: Plant Phenomics (under review)

This script implements a multi‑task Gaussian Process regressor using a combined
kernel (RBF + WhiteKernel). It jointly predicts four alfalfa quality traits
(N, CP, ADF, NDF) from multi‑temporal UAV multispectral features.

=============================================================================
Data & Model Availability
=============================================================================
The pre‑trained model weights (if any) and inference code are available at:
    https://huggingface.co/PheniX-Lab/GAI-Estimation/tree/main

The raw spectral data and ground‑truth quality data used in the original study are
not publicly available due to ongoing collaborative projects, but are available from
the corresponding author on reasonable request.

=============================================================================
How to use this code
=============================================================================
1. Prepare your data:
   - X: numpy array of shape (n_samples, n_features)
         Features should be the 27‑dimensional concatenation of the 9 orthogonal
         spectral features (Table 2 of the paper) for three growth stages.
   - y: numpy array of shape (n_samples, 4)
         Columns in order: N (%), CP (%), ADF (%), NDF (%).

2. Standardize features (highly recommended for GPR).

3. Call `train_mt_gpr()` to train the model.

4. Evaluate predictions using the provided `calc_metrics()` function.

Requirements:
   Python 3.8+, numpy, pandas, scikit‑learn
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def calc_metrics(y_true, y_pred):
    """
    Compute regression metrics: R², RMSE, MAE, RPD.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if np.std(y_pred) > 1e-6 and np.std(y_true) > 1e-6:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = 0.0
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    sd_y = np.std(y_true, ddof=1)
    rpd = sd_y / rmse if rmse > 1e-6 else np.inf
    return r2, rmse, mae, rpd


def train_mt_gpr(X_train, y_train, X_test=None, y_test=None,
                 kernel=None, alpha=0.1, random_state=42, normalize_y=True):
    """
    Train a Multi‑Task Gaussian Process regressor.

    Parameters
    ----------
    X_train : array-like, shape (n_train, n_features)
        Training features.
    y_train : array-like, shape (n_train, 4)
        Training targets (N, CP, ADF, NDF).
    X_test, y_test : optional, for evaluation.
    kernel : sklearn.gaussian_process.kernels.Kernel, optional.
        Default: RBF() + WhiteKernel().
    alpha : float, noise level.
    random_state : int.
    normalize_y : bool, whether to normalize target values.

    Returns
    -------
    model : GaussianProcessRegressor
        Trained model.
    pred_train : numpy.ndarray
        Predictions on training set.
    pred_test : numpy.ndarray or None
        Predictions on test set if provided.
    metrics : dict
        Dictionary containing metrics for training and test sets.
    """
    if kernel is None:
        kernel = RBF() + WhiteKernel()

    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        random_state=random_state,
        normalize_y=normalize_y
    )
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    metrics_train = {}
    for i, target in enumerate(['N', 'CP', 'ADF', 'NDF']):
        r2, rmse, mae, rpd = calc_metrics(y_train[:, i], pred_train[:, i])
        metrics_train[target] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'RPD': rpd}

    if X_test is not None and y_test is not None:
        pred_test = model.predict(X_test)
        metrics_test = {}
        for i, target in enumerate(['N', 'CP', 'ADF', 'NDF']):
            r2, rmse, mae, rpd = calc_metrics(y_test[:, i], pred_test[:, i])
            metrics_test[target] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'RPD': rpd}
    else:
        pred_test = None
        metrics_test = None

    return model, pred_train, pred_test, {'train': metrics_train, 'test': metrics_test}


# ============================================================================
# Example usage (replace with your own data)
# ============================================================================
if __name__ == "__main__":
    print("=== Multi‑Task Gaussian Process Regression (MT-GPR) ===")
    print("Please replace the synthetic data below with your own data loading.")
    print("Expected data shapes: X (n_samples, n_features), y (n_samples, 4).")
    print()

    # ------------------------------------------------------------------------
    # [REPLACE THIS BLOCK] Load your own data
    # ------------------------------------------------------------------------
    # Example: Generate synthetic data (do not use in real application)
    np.random.seed(42)
    n_samples = 300
    n_features = 27  # 3 stages × 9 features
    X_demo = np.random.randn(n_samples, n_features)
    y_demo = np.zeros((n_samples, 4))
    y_demo[:, 0] = 3.5 + 0.5 * X_demo[:, 0] + 0.2 * np.random.randn(n_samples)   # N
    y_demo[:, 1] = 22.0 + 3.0 * X_demo[:, 1] + 1.0 * np.random.randn(n_samples)  # CP
    y_demo[:, 2] = 26.0 - 2.0 * X_demo[:, 2] + 1.0 * np.random.randn(n_samples)  # ADF
    y_demo[:, 3] = 40.0 - 3.0 * X_demo[:, 3] + 1.5 * np.random.randn(n_samples)  # NDF
    # ------------------------------------------------------------------------

    # Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(X_demo, y_demo, test_size=0.25, random_state=42)

    # Standardization (critical for GPR)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # Train model
    model, pred_tr, pred_te, metrics = train_mt_gpr(X_tr, y_tr, X_te, y_te)

    # Print test set metrics
    print("\nTest set performance:")
    for target, m in metrics['test'].items():
        print(f"  {target}: R² = {m['R2']:.3f}, RMSE = {m['RMSE']:.3f}, RPD = {m['RPD']:.3f}")

    # Optional: save model (scikit-learn models can be saved with joblib)
    # import joblib
    # joblib.dump(model, 'mt_gpr_model.pkl')