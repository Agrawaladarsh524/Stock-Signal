"""
Predicting the Direction of Reliance Industries Stock Prices Using Random Forest

Methodology adapted from:
Khaidem, Luckyson, Snehanshu Saha, and Sudeepa Roy Dey.
"Predicting the direction of stock market prices using random forest."
arXiv preprint arXiv:1605.00003 (2016).

This script applies the same Random Forest approach to Reliance Industries
(NSE: RELIANCE) historical data from 2016 to 2025.
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys

# Make results reproducible
np.random.seed(42)
random.seed(42)

# Import the technical indicators module from the same directory
import pandas_techinal_indicators as ta

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    confusion_matrix,
    recall_score,
    accuracy_score,
)


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_reliance_data(filepath="RELIANCE_historical_final_data.csv"):
    """Load Reliance Industries historical stock data.

    The CSV has columns: Date, Adj Close, Close, High, Low, Open, Volume
    We drop 'Date' and 'Adj Close' to match the format expected by the
    technical indicators module (Open, High, Low, Close, Volume).
    """
    df = pd.read_csv(filepath)
    # Drop Date and Adj Close (same preprocessing as the original AAPL analysis)
    del df["Date"]
    del df["Adj Close"]
    print(f"Loaded {len(df)} rows of Reliance stock data.")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print()
    return df


# =============================================================================
# 2. EXPONENTIAL SMOOTHING
# =============================================================================

def get_exp_preprocessing(df, alpha=0.9):
    """Apply exponential smoothing to the data.

    The authors don't specify alpha, so we use 0.9 (same as the AAPL analysis).
    """
    edata = df.ewm(alpha=alpha).mean()
    return edata


# =============================================================================
# 3. FEATURE EXTRACTION — TECHNICAL INDICATORS
# =============================================================================

def feature_extraction(data):
    """Extract technical indicator features from OHLCV data.

    Uses multiple window sizes (5, 14, 26, 44, 66) for each indicator:
    - Relative Strength Index (RSI)
    - Stochastic Oscillator %D
    - Accumulation/Distribution ROC
    - Average True Range (ATR)
    - Momentum
    - Money Flow Index (MFI)
    - Rate of Change (ROC)
    - On-Balance Volume (OBV)
    - Commodity Channel Index (CCI)
    - Ease of Movement (EoM)
    - TRIX
    - Vortex Indicator

    Plus EMA ratios (5, 14, 21, 50) and MACD (12/26).
    """
    for x in [5, 14, 26, 44, 66]:
        data = ta.relative_strength_index(data, n=x)
        data = ta.stochastic_oscillator_d(data, n=x)
        data = ta.accumulation_distribution(data, n=x)
        data = ta.average_true_range(data, n=x)
        data = ta.momentum(data, n=x)
        data = ta.money_flow_index(data, n=x)
        data = ta.rate_of_change(data, n=x)
        data = ta.on_balance_volume(data, n=x)
        data = ta.commodity_channel_index(data, n=x)
        data = ta.ease_of_movement(data, n=x)
        data = ta.trix(data, n=x)
        data = ta.vortex_indicator(data, n=x)

    data["ema50"] = data["Close"] / data["Close"].ewm(50).mean()
    data["ema21"] = data["Close"] / data["Close"].ewm(21).mean()
    data["ema14"] = data["Close"] / data["Close"].ewm(14).mean()
    data["ema5"] = data["Close"] / data["Close"].ewm(5).mean()

    # Williams %R is missing from the indicators library
    data = ta.macd(data, n_fast=12, n_slow=26)

    del data["Open"]
    del data["High"]
    del data["Low"]
    del data["Volume"]

    return data


def compute_prediction_int(df, n):
    """Compute binary prediction target.

    Returns 1 if the Close price n days later is >= current Close, else 0.
    """
    pred = df.shift(-n)["Close"] >= df["Close"]
    pred = pred.iloc[:-n]
    return pred.astype(int)


def prepare_data(df, horizon):
    """Prepare feature matrix and target variable.

    1. Extract technical indicator features
    2. Drop rows with NaN (from indicator warm-up periods)
    3. Compute binary prediction target for the given horizon
    4. Drop Close column (it's the basis for the target, not a feature)
    """
    data = feature_extraction(df).dropna().iloc[:-horizon]
    data["pred"] = compute_prediction_int(data, n=horizon)
    del data["Close"]
    return data.dropna()


# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("  RELIANCE INDUSTRIES — Stock Direction Prediction Using Random Forest")
    print("=" * 70)
    print()

    # --- Load Data ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "RELIANCE_historical_final_data.csv")

    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset not found at {csv_path}")
        sys.exit(1)

    reliance = load_reliance_data(csv_path)

    # --- Exponential Smoothing ---
    print("Applying exponential smoothing (alpha=0.9)...")
    sreliance = get_exp_preprocessing(reliance)
    print("Smoothed data (first 5 rows):")
    print(sreliance.head())
    print()

    # --- Feature Extraction & Target ---
    horizon = 10  # 10-day prediction horizon (same as original AAPL analysis)
    print(f"Preparing data with prediction horizon = {horizon} days...")
    data = prepare_data(sreliance, horizon)

    y = data["pred"]
    features = [x for x in data.columns if x not in ["gain", "pred"]]
    X = data[features]

    print(f"Total samples after feature extraction: {len(X)}")
    print(f"Number of features: {len(features)}")
    print()

    # --- Train/Test Split (chronological, no shuffling) ---
    train_size = 2 * len(X) // 3

    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size:  {len(X_test)}")
    print(f"Train period class distribution: {dict(y_train.value_counts())}")
    print(f"Test period class distribution:  {dict(y_test.value_counts())}")
    print()

    # --- Random Forest Training ---
    print("Training Random Forest classifier (n_estimators=65)...")
    rf = RandomForestClassifier(n_jobs=-1, n_estimators=65, random_state=42)
    rf.fit(X_train, y_train.values.ravel())
    print("Training complete!")
    print()

    # --- Evaluation ---
    pred = rf.predict(X_test)
    precision = precision_score(y_pred=pred, y_true=y_test)
    recall = recall_score(y_pred=pred, y_true=y_test)
    f1 = f1_score(y_pred=pred, y_true=y_test)
    accuracy = accuracy_score(y_pred=pred, y_true=y_test)
    conf_matrix = confusion_matrix(y_pred=pred, y_true=y_test)

    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print()
    print("  Confusion Matrix:")
    print(f"  {conf_matrix}")
    print()
    print(f"  (Prediction horizon: {horizon} days)")
    print("=" * 70)
    print()

    # --- Visualization ---
    plt.rcParams["figure.figsize"] = (10, 6)

    # Plot 1: Feature Importances (top 20)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # top 20

    plt.figure()
    plt.title("Reliance Stock — Top 20 Feature Importances (Random Forest)")
    plt.bar(range(20), importances[indices], align="center", color="steelblue")
    plt.xticks(range(20), [features[i] for i in indices], rotation=90, fontsize=8)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "reliance_feature_importances.png"), dpi=150)
    print("Saved: reliance_feature_importances.png")

    # Plot 2: Prediction vs Actual on test set
    plt.figure()
    plt.title(f"Reliance Stock — Predicted vs Actual Direction ({horizon}-day horizon)")
    plt.plot(range(len(y_test)), y_test.values, label="Actual", alpha=0.7, linewidth=1)
    plt.plot(range(len(pred)), pred, label="Predicted", alpha=0.7, linewidth=1, linestyle="--")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Direction (0=Down, 1=Up)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "reliance_prediction_comparison.png"), dpi=150)
    print("Saved: reliance_prediction_comparison.png")

    # Plot 3: Closing Price Chart
    reliance_full = pd.read_csv(csv_path)
    plt.figure()
    plt.title("Reliance Industries — Historical Closing Price (2016–2025)")
    plt.plot(pd.to_datetime(reliance_full["Date"]), reliance_full["Close"],
             color="darkblue", linewidth=0.8)
    plt.xlabel("Date")
    plt.ylabel("Close Price (INR)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "reliance_closing_price.png"), dpi=150)
    print("Saved: reliance_closing_price.png")

    print("\nDone! All plots saved to the project directory.")


if __name__ == "__main__":
    main()
