"""
data_generator.py
Generates realistic synthetic infrastructure metrics dataset
with injected anomaly patterns: traffic spikes, memory leaks,
latency bursts, and normal baseline load.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── config ───────────────────────────────────────────────────────────────────
N_POINTS    = 2_000   # total time-steps
FREQ_SEC    = 10      # seconds between samples
ANOMALY_PCT = 0.08    # ~8 % of points are anomalies


def generate_dataset(n: int = N_POINTS) -> pd.DataFrame:
    """
    Build a DataFrame of simulated infrastructure metrics.

    Patterns injected
    -----------------
    - Normal baseline  : Gaussian noise around healthy operating levels.
    - Traffic spike    : sudden burst in cpu + request_rate.
    - Memory leak      : slow monotonic RAM increase over a window.
    - Latency burst    : isolated spike in network_latency.
    """
    timestamps = [datetime(2024, 1, 1) + timedelta(seconds=i * FREQ_SEC)
                  for i in range(n)]

    # ── baseline metrics (normal, healthy) ───────────────────────────────────
    cpu      = np.random.normal(35, 8,  n).clip(5,  100)
    ram      = np.random.normal(55, 10, n).clip(10, 100)
    disk_io  = np.random.normal(40, 12, n).clip(0,  100)
    latency  = np.random.normal(80, 15, n).clip(10, 500)
    req_rate = np.random.normal(200, 40, n).clip(10, 2000)

    # ── anomaly mask (starts all-normal) ─────────────────────────────────────
    labels = np.zeros(n, dtype=int)

    n_anomalies = int(n * ANOMALY_PCT)
    anomaly_starts = np.random.choice(range(50, n - 50), n_anomalies, replace=False)

    for start in anomaly_starts:
        kind = np.random.choice(["traffic_spike", "memory_leak", "latency_burst"])
        end  = min(start + np.random.randint(5, 20), n - 1)

        if kind == "traffic_spike":
            cpu[start:end]      += np.random.uniform(40, 60)
            req_rate[start:end] += np.random.uniform(600, 1000)

        elif kind == "memory_leak":
            leak = np.linspace(20, 45, end - start)
            ram[start:end] += leak

        else:  # latency_burst
            latency[start:end] += np.random.uniform(250, 450)

        labels[start:end] = 1

    # ── clip back to physical limits ─────────────────────────────────────────
    cpu      = cpu.clip(0, 100)
    ram      = ram.clip(0, 100)
    latency  = latency.clip(0, 1000)
    req_rate = req_rate.clip(0, 5000)

    df = pd.DataFrame({
        "timestamp"       : timestamps,
        "cpu_usage"       : np.round(cpu,      2),
        "ram_usage"       : np.round(ram,      2),
        "disk_io"         : np.round(disk_io,  2),
        "network_latency" : np.round(latency,  2),
        "request_rate"    : np.round(req_rate, 2),
        "anomaly_label"   : labels,
    })

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("infrastructure_metrics.csv", index=False)
    print(f"[DataGen] Saved {len(df):,} rows  |  "
          f"Anomalies: {df['anomaly_label'].sum()} "
          f"({df['anomaly_label'].mean()*100:.1f} %)")
    print(df.head())
