"""
train_model.py
Trains an Isolation Forest on NORMAL data only (unsupervised),
then evaluates detection quality and persists the model + scaler.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ── import data generator so we can run standalone ──────────────────────────
from data_generator import generate_dataset

# ── feature columns used for training ───────────────────────────────────────
FEATURES = ["cpu_usage", "ram_usage", "disk_io", "network_latency", "request_rate"]

# ── Isolation Forest hyper-parameters ───────────────────────────────────────
IF_PARAMS = dict(
    n_estimators=200,        # more trees → more stable scores
    max_samples="auto",
    contamination=0.08,      # expected anomaly fraction (matches our generator)
    random_state=42,
    n_jobs=-1,
)


def train(df: pd.DataFrame):
    """
    1. Scale features.
    2. Train Isolation Forest on NORMAL rows only (label == 0).
    3. Evaluate on full dataset to report precision / recall.
    4. Save model + scaler to disk.
    """
    X_all = df[FEATURES].values
    y_all = df["anomaly_label"].values

    # ── fit scaler on normal data only ───────────────────────────────────────
    normal_mask = (y_all == 0)
    scaler = StandardScaler()
    scaler.fit(X_all[normal_mask])
    X_scaled = scaler.transform(X_all)

    # ── train Isolation Forest on normal data only ────────────────────────────
    print("[Train] Fitting Isolation Forest …")
    model = IsolationForest(**IF_PARAMS)
    model.fit(X_scaled[normal_mask])

    # ── predict on full dataset ───────────────────────────────────────────────
    # IF returns +1 (normal) / -1 (anomaly) → convert to 0 / 1
    raw_pred   = model.predict(X_scaled)
    y_pred     = np.where(raw_pred == -1, 1, 0)

    # anomaly_score: higher = more anomalous  (we negate decision_function)
    scores_raw = model.decision_function(X_scaled)   # higher = more normal
    anomaly_scores = -scores_raw                     # flip: higher = more anomalous

    # ── evaluation ───────────────────────────────────────────────────────────
    print("\n[Eval] Classification Report:")
    print(classification_report(y_all, y_pred, target_names=["Normal", "Anomaly"]))
    print("[Eval] Confusion Matrix:")
    print(confusion_matrix(y_all, y_pred))

    # ── save artifacts ────────────────────────────────────────────────────────
    joblib.dump(model,  "isolation_forest.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\n[Train] Model saved  → isolation_forest.pkl")
    print("[Train] Scaler saved → scaler.pkl")

    # ── also save scored dataset for dashboard use ───────────────────────────
    df_out = df.copy()
    df_out["anomaly_score"] = anomaly_scores
    df_out["predicted"]     = y_pred
    df_out.to_csv("infrastructure_metrics.csv", index=False)
    print("[Train] Scored dataset → infrastructure_metrics.csv")

    return model, scaler


if __name__ == "__main__":
    print("[DataGen] Generating dataset …")
    df = generate_dataset()
    train(df)
