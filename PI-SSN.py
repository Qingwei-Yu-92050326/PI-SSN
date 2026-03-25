# -*- coding: utf-8 -*-
"""
PI-SSN: Physics-Informed Sparse Shallow Network

This repository contains the official implementation of PI-SSN, a model for predicting
alfalfa quality traits (N, CP, ADF, NDF) from multi‑temporal UAV multispectral data.

======================================================================

The raw spectral data and ground‑truth quality data used in this study are not
publicly available due to ongoing collaborative projects, but are available from
the corresponding author on reasonable request.

=============================================================================
How to use this code
=============================================================================
1. Prepare your own data:
   - Each sample should contain reflectance values for the 10 bands (405,430,450,550,
     560,570,650,685,710,850 nm) for three growth stages (T1, T2, T3).
   - The target variables are four quality traits: N (%), CP (%), ADF (%), NDF (%).

2. Call `extract_orthogonal_features()` on each growth stage to compute the 9
   orthogonal features defined in Table 2 of the paper.

3. Concatenate features from the three stages into a single 27‑dimensional vector
   per sample (3 stages × 9 features).

4. Use the `PI_SSN` class to define the model, then train it with
   `train_pi_ssn()` or load a pre‑trained checkpoint.

5. For inference, see the example at the end of the script.

Requirements:
   Python 3.8+, PyTorch 1.10+, numpy, pandas, scikit‑learn
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. Feature extraction (based on Table 2 of the paper)
# ============================================================================
def extract_orthogonal_features(df_bands):
    """
    Compute the nine orthogonal spectral features from raw reflectance data.

    Parameters
    ----------
    df_bands : pandas.DataFrame
        Reflectance values for the 10 bands (405,430,450,550,560,570,650,685,710,850 nm).
        Index: sample ID; columns: wavelength (nm).

    Returns
    -------
    pandas.DataFrame
        DataFrame with nine feature columns:
        Bio_F1, Bio_F2, Bio_F3, Fib_F1, Fib_F2, Fib_F3, Fib_F4, Fib_F5, Fib_F6.
    """
    # Logarithmic transformation (Beer–Lambert law)
    df_log = np.log(1 / df_bands)

    feats = pd.DataFrame(index=df_bands.index)

    # ----- Biochemical traits (N, CP) -----
    # Bio_F1: absorption‑scattering decoupling (3B-Pigment, 560,710,405) [OR space]
    feats['Bio_F1'] = (1 / df_bands[560] - 1 / df_bands[710]) * df_bands[405]

    # Bio_F2: 4B-DD (405,570,560,710) [OR space]
    feats['Bio_F2'] = (df_bands[405] - df_bands[570]) / (df_bands[560] - df_bands[710])

    # Bio_F3: 4B-DD (405,550,570,650) [Log space]
    feats['Bio_F3'] = (df_log[405] - df_log[550]) / (df_log[570] - df_log[650])

    # ----- Structural traits (ADF, NDF) -----
    # Fib_F1: 4B-DD (430,560,450,570) [OR space]
    feats['Fib_F1'] = (df_bands[430] - df_bands[560]) / (df_bands[450] - df_bands[570])

    # Fib_F2: 3B-VARI (405,550,850) [Log space]
    feats['Fib_F2'] = (df_log[405] - df_log[550]) / (df_log[405] + df_log[550] - df_log[850])

    # Fib_F3: 4B-DD (405,560,450,850) [Log space]
    feats['Fib_F3'] = (df_log[405] - df_log[560]) / (df_log[450] - df_log[850])

    # Fib_F4: 4B-DD (570,710,405,550) [OR space]
    feats['Fib_F4'] = (df_bands[570] - df_bands[710]) / (df_bands[405] - df_bands[550])

    # Fib_F5: 3B-MTCI (550,710,405) [OR space]
    feats['Fib_F5'] = (df_bands[550] - df_bands[710]) / (df_bands[710] - df_bands[405])

    # Fib_F6: 4B-DD (450,850,405,550) [Log space]
    feats['Fib_F6'] = (df_log[450] - df_log[850]) / (df_log[405] - df_log[550])

    # Replace infinities and NaNs
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0)
    return feats


def build_temporal_features(df_bands_list, time_points):
    """
    Concatenate features from multiple growth stages into a single matrix.

    Parameters
    ----------
    df_bands_list : list of pandas.DataFrame
        Each element contains reflectance data for one growth stage.
    time_points : list of str
        Names of the growth stages (e.g., ['T1', 'T2', 'T3']).

    Returns
    -------
    pandas.DataFrame
        Features for all stages, columns named as <feature>_<time_point>.
    """
    all_feats = []
    for i, df_bands in enumerate(df_bands_list):
        feats = extract_orthogonal_features(df_bands)
        feats = feats.add_suffix(f'_{time_points[i]}')
        all_feats.append(feats)
    return pd.concat(all_feats, axis=1)


# ============================================================================
# 2. PI‑SSN model definition
# ============================================================================
class TemporalAttention(nn.Module):
    """Temporal attention module that adaptively weights growth stages."""
    def __init__(self, input_dim=9, hidden_dim=8):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x shape: (batch, n_time, feat_per_time)
        attn_weights = self.attention(x)          # (batch, n_time, 1)
        context = torch.sum(x * attn_weights, dim=1)  # (batch, feat_per_time)
        return context, attn_weights


class PI_SSN(nn.Module):
    """
    Physics-Informed Sparse Shallow Network.

    Features:
        - Two-stream architecture: biochemical stream (Tanh, no L1) and structural
          stream (LeakyReLU + L1 sparsity).
        - Temporal attention decoupling: different growth‑stage weights for
          biochemical and structural traits.
    """
    def __init__(self, n_time=3, feat_per_time=9, n_bio=2, n_struc=2):
        super().__init__()
        self.n_time = n_time
        self.feat_per_time = feat_per_time
        total_feat = n_time * feat_per_time

        # Temporal attention (separate for each stream)
        self.attn_bio = TemporalAttention(input_dim=feat_per_time)
        self.attn_struc = TemporalAttention(input_dim=feat_per_time)

        # Biochemical stream (smooth, no L1 regularization)
        self.bio_stream = nn.Sequential(
            nn.Linear(total_feat + feat_per_time, 16),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(16, n_bio)
        )

        # Structural stream (sparse, L1 regularization applied during training)
        self.struc_stream = nn.Sequential(
            nn.Linear(total_feat + feat_per_time, 12),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(12, n_struc)
        )

    def forward(self, x_flat, return_attn=False):
        """
        Parameters
        ----------
        x_flat : torch.Tensor
            Flattened temporal features, shape (batch, n_time * feat_per_time).
        return_attn : bool
            If True, also return attention weights (useful for visualisation).

        Returns
        -------
        torch.Tensor
            Predictions, shape (batch, n_bio + n_struc).
        torch.Tensor (optional)
            Attention weights for biochemical and structural streams.
        """
        batch_size = x_flat.shape[0]
        # Reshape to (batch, n_time, feat_per_time)
        x_time = x_flat.view(batch_size, self.n_time, self.feat_per_time)

        # Obtain temporal context
        context_bio, attn_bio = self.attn_bio(x_time)
        context_struc, attn_struc = self.attn_struc(x_time)

        # Concatenate original features with temporal context
        feat_bio = torch.cat([x_flat, context_bio], dim=1)
        feat_struc = torch.cat([x_flat, context_struc], dim=1)

        # Forward through streams
        bio_preds = self.bio_stream(feat_bio)
        struc_preds = self.struc_stream(feat_struc)

        preds = torch.cat([bio_preds, struc_preds], dim=1)

        if return_attn:
            return preds, attn_bio, attn_struc
        return preds


# ============================================================================
# 3. Physics‑informed loss function
# ============================================================================
def physics_informed_loss(model, preds, targets, lambda_phy=0.5, lambda_l1=0.01):
    """
    Combined loss with physical constraints.

    Parameters
    ----------
    model : PI_SSN
        The model instance (used to access structural stream for L1 penalty).
    preds : torch.Tensor
        Predictions, shape (batch, 4) in order [N, CP, ADF, NDF].
    targets : torch.Tensor
        Ground‑truth values, same shape.
    lambda_phy : float
        Weight for the physical constraint term.
    lambda_l1 : float
        Weight for the L1 sparsity penalty.

    Returns
    -------
    torch.Tensor
        Total loss.
    """
    # Base regression loss
    loss_N = nn.MSELoss()(preds[:, 0], targets[:, 0])
    loss_CP = nn.MSELoss()(preds[:, 1], targets[:, 1])
    loss_ADF = nn.SmoothL1Loss()(preds[:, 2], targets[:, 2])
    loss_NDF = nn.SmoothL1Loss()(preds[:, 3], targets[:, 3])
    mse_total = loss_N + loss_CP + loss_ADF + loss_NDF

    # Physical constraint: CP and NDF should be negatively correlated
    cp_pred = preds[:, 1]
    ndf_pred = preds[:, 3]
    cov_cp_ndf = torch.mean((cp_pred - cp_pred.mean()) * (ndf_pred - ndf_pred.mean()))
    physics_penalty = nn.functional.softplus(cov_cp_ndf)   # penalty only when positive

    # L1 sparsity penalty (only applied to the structural stream)
    l1_penalty = 0.0
    for param in model.struc_stream.parameters():
        l1_penalty += torch.sum(torch.abs(param))

    return mse_total + lambda_phy * physics_penalty + lambda_l1 * l1_penalty


# ============================================================================
# 4. Training routine
# ============================================================================
def train_pi_ssn(X_train, y_train, X_test, y_test,
                 epochs=600, lr=0.008, batch_size=32, patience=30):
    """
    Train the PI‑SSN model.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training features, shape (n_train, n_time * feat_per_time).
    y_train : numpy.ndarray
        Training labels, shape (n_train, 4) [N, CP, ADF, NDF].
    X_test : numpy.ndarray
        Test features.
    y_test : numpy.ndarray
        Test labels (if None, only training is performed).
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    batch_size : int
        Batch size.
    patience : int
        Patience for ReduceLROnPlateau scheduler.

    Returns
    -------
    model : PI_SSN
        Trained model (best weights loaded).
    pred_train : numpy.ndarray
        Predictions on the training set (original scale).
    pred_test : numpy.ndarray
        Predictions on the test set (original scale) or None.
    history : list
        Training loss per epoch.
    """
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test) if X_test is not None else None

    # Data loader
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PI_SSN()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=patience)

    best_loss = float('inf')
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            # Small noise for robustness (optional)
            if model.training:
                batch_X = batch_X + torch.randn_like(batch_X) * 0.03
            preds = model(batch_X)
            # Gradually increase physical constraint strength
            current_lambda = min(1.0, epoch / 150.0)
            loss = physics_informed_loss(model, preds, batch_y,
                                         lambda_phy=current_lambda,
                                         lambda_l1=0.05)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step(epoch_loss)
        history.append(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = model.state_dict().copy()

    # Load best weights
    model.load_state_dict(best_state)

    # Predictions
    model.eval()
    with torch.no_grad():
        pred_train = model(X_train_t).numpy()
        pred_test = model(X_test_t).numpy() if X_test is not None else None

    return model, pred_train, pred_test, history


# ============================================================================
# 5. Example usage (replace with your own data loading)
# ============================================================================
if __name__ == "__main__":
    # ------------------------------------------------------------------------
    # 5.1 Load your own data (this part must be customised)
    # ------------------------------------------------------------------------
    # Here you should read your data, compute the orthogonal features for each
    # growth stage, and concatenate them into a 27‑dimensional feature vector
    # per sample. The following code shows the expected structure.
    #
    # Example:
    #   df_T1 = pd.read_excel('stage1_reflectance.xlsx')
    #   df_T2 = pd.read_excel('stage2_reflectance.xlsx')
    #   df_T3 = pd.read_excel('stage3_reflectance.xlsx')
    #   df_target = pd.read_excel('quality_targets.xlsx')
    #
    #   feats_T1 = extract_orthogonal_features(df_T1).add_suffix('_T1')
    #   feats_T2 = extract_orthogonal_features(df_T2).add_suffix('_T2')
    #   feats_T3 = extract_orthogonal_features(df_T3).add_suffix('_T3')
    #   X = pd.concat([feats_T1, feats_T2, feats_T3], axis=1).values
    #   y = df_target[['N', 'CP', 'ADF', 'NDF']].values
    #
    #   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    #   scaler_X = StandardScaler()
    #   scaler_y = StandardScaler()
    #   X_train = scaler_X.fit_transform(X_train)
    #   X_test = scaler_X.transform(X_test)
    #   y_train = scaler_y.fit_transform(y_train)
    #
    #   model, pred_train, pred_test, _ = train_pi_ssn(X_train, y_train, X_test, None)
    #
    # ------------------------------------------------------------------------
    # For demonstration, we simulate a small dataset (replace with real data)
    # ------------------------------------------------------------------------
    print("=== PI‑SSN Example Usage ===")
    print("Please replace the synthetic data below with your own data loading.")
    print("The code expects X (features) and y (targets) to be numpy arrays.")
    print()

    # --- Replace this block with your actual data loading ---
    np.random.seed(42)
    n_samples = 200
    n_features = 27      # 3 stages × 9 features
    X_demo = np.random.randn(n_samples, n_features)
    y_demo = np.zeros((n_samples, 4))
    # Simulate plausible relationships (just for demonstration)
    y_demo[:, 0] = 3.5 + 0.5 * X_demo[:, 0] + 0.2 * np.random.randn(n_samples)   # N
    y_demo[:, 1] = 22.0 + 3.0 * X_demo[:, 1] + 1.0 * np.random.randn(n_samples)  # CP
    y_demo[:, 2] = 26.0 - 2.0 * X_demo[:, 2] + 1.0 * np.random.randn(n_samples)  # ADF
    y_demo[:, 3] = 40.0 - 3.0 * X_demo[:, 3] + 1.5 * np.random.randn(n_samples)  # NDF
    # ---------------------------------------------------------

    # Train / test split
    X_tr, X_te, y_tr, y_te = train_test_split(X_demo, y_demo, test_size=0.25, random_state=42)

    # Standardisation
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_tr = scaler_X.fit_transform(X_tr)
    X_te = scaler_X.transform(X_te)
    y_tr_scaled = scaler_y.fit_transform(y_tr)

    # Train model
    print("Training PI‑SSN ...")
    model, pred_tr_scaled, pred_te_scaled, _ = train_pi_ssn(X_tr, y_tr_scaled, X_te, None)

    # Inverse transform predictions
    pred_tr = scaler_y.inverse_transform(pred_tr_scaled)
    pred_te = scaler_y.inverse_transform(pred_te_scaled)

    # Simple evaluation
    from sklearn.metrics import r2_score, mean_squared_error
    targets = ['N', 'CP', 'ADF', 'NDF']
    print("\nTest set performance:")
    for i, name in enumerate(targets):
        r2 = r2_score(y_te[:, i], pred_te[:, i])
        rmse = np.sqrt(mean_squared_error(y_te[:, i], pred_te[:, i]))
        print(f"  {name}: R² = {r2:.3f}, RMSE = {rmse:.3f}")

    # ------------------------------------------------------------------------
    # 5.2 Save/load model weights
    # ------------------------------------------------------------------------
    # Save the trained weights
    torch.save(model.state_dict(), 'PI_SSN_best.pth')
    print("\nModel weights saved to PI_SSN_best.pth")

    # ------------------------------------------------------------------------
    # 5.3 Inference with pre‑trained weights (example)
    # ------------------------------------------------------------------------
    print("\nLoading pre‑trained weights for inference ...")
    # Re‑create the model and load weights
    inference_model = PI_SSN()
    inference_model.load_state_dict(torch.load('PI_SSN_best.pth'))
    inference_model.eval()

    # Prepare new data (replace with your actual data)
    # The data must already be scaled using the same StandardScaler used during training.
    X_new = X_te[:5]   # take 5 test samples as an example
    X_new_tensor = torch.FloatTensor(X_new)

    with torch.no_grad():
        pred_new_scaled = inference_model(X_new_tensor).numpy()
        pred_new = scaler_y.inverse_transform(pred_new_scaled)

    print("Predictions for new samples (N, CP, ADF, NDF):")
    print(pred_new)
