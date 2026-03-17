"""
monitor.py
Real-time monitoring simulation loop.
- Loads trained model + scaler.
- Streams rows from the scored CSV one at a time (simulating live telemetry).
- Prints colour-coded alerts to the terminal.
- Also exposes a generator function used by the Streamlit dashboard.
"""

import time
import joblib
import numpy as np
import pandas as pd

from agent import DevOpsAgent

# ── constants ────────────────────────────────────────────────────────────────
FEATURES        = ["cpu_usage", "ram_usage", "disk_io", "network_latency", "request_rate"]
ANOMALY_THRESH  = 0.05        # anomaly_score above this triggers an alert
STREAM_DELAY    = 0.5         # seconds between "live" samples (CLI mode)

# ANSI colour helpers for terminal output
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RESET  = "\033[0m"


def load_artifacts():
    """Load model and scaler from disk."""
    model  = joblib.load("isolation_forest.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


def diagnose_anomaly(row: pd.Series) -> list[str]:
    """
    Compare each metric against rough 'healthy' thresholds
    and return a list of flagged metric names.
    """
    flags = []
    if row["cpu_usage"]       > 80:  flags.append("CPU")
    if row["ram_usage"]       > 85:  flags.append("RAM")
    if row["network_latency"] > 200: flags.append("Latency")
    if row["disk_io"]         > 85:  flags.append("DiskIO")
    if row["request_rate"]    > 800: flags.append("RequestRate")
    return flags or ["(composite anomaly)"]


def score_row(row: pd.Series, model, scaler) -> tuple[int, float]:
    """
    Run inference on a single metric row.
    Returns (predicted_label, anomaly_score).
    """
    X = scaler.transform([row[FEATURES].values])
    pred  = model.predict(X)[0]               # +1 normal / -1 anomaly
    score = -model.decision_function(X)[0]    # higher = more anomalous
    label = 1 if pred == -1 else 0
    return label, float(score)


# ── streaming generator (used by Streamlit) ──────────────────────────────────
def stream_metrics(df: pd.DataFrame, model, scaler):
    """
    Yield one enriched row dict at a time.
    The dashboard calls next() on this generator every refresh cycle.
    """
    agent = DevOpsAgent()
    for _, row in df.iterrows():
        label, score = score_row(row, model, scaler)
        flags        = diagnose_anomaly(row) if label == 1 else []
        actions      = agent.recommend(row, flags) if label == 1 else []
        yield {
            "timestamp"       : row["timestamp"],
            "cpu_usage"       : row["cpu_usage"],
            "ram_usage"       : row["ram_usage"],
            "disk_io"         : row["disk_io"],
            "network_latency" : row["network_latency"],
            "request_rate"    : row["request_rate"],
            "anomaly_label"   : int(label),
            "anomaly_score"   : round(score, 4),
            "flags"           : flags,
            "actions"         : actions,
        }


# ── CLI real-time simulation ──────────────────────────────────────────────────
def run_cli():
    print("[Monitor] Loading model …")
    model, scaler = load_artifacts()
    df = pd.read_csv("infrastructure_metrics.csv", parse_dates=["timestamp"])
    agent = DevOpsAgent()

    print(f"[Monitor] Starting real-time simulation  "
          f"({STREAM_DELAY}s per sample) — press Ctrl+C to stop\n")

    for _, row in df.iterrows():
        label, score = score_row(row, model, scaler)

        ts = row["timestamp"].strftime("%H:%M:%S")

        if label == 1:
            flags   = diagnose_anomaly(row)
            actions = agent.recommend(row, flags)
            flag_str = ", ".join(flags)

            print(f"{RED}⚠  ANOMALY  {ts}  score={score:.3f}  "
                  f"flags=[{flag_str}]{RESET}")
            print(f"   CPU={row['cpu_usage']:.1f}%  "
                  f"RAM={row['ram_usage']:.1f}%  "
                  f"Lat={row['network_latency']:.0f}ms  "
                  f"ReqRate={row['request_rate']:.0f}")
            for a in actions:
                print(f"   {YELLOW}→ {a}{RESET}")
            print()
        else:
            print(f"{GREEN}✓  normal   {ts}  score={score:.3f}  "
                  f"CPU={row['cpu_usage']:.1f}%  "
                  f"RAM={row['ram_usage']:.1f}%  "
                  f"Lat={row['network_latency']:.0f}ms{RESET}")

        time.sleep(STREAM_DELAY)


if __name__ == "__main__":
    run_cli()
