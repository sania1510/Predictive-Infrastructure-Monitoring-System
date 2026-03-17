# 🖥️ Predictive Infrastructure Monitoring System

> ML-Powered Anomaly Detection for DevOps Teams


## 📌 Overview

A complete prototype that simulates how DevOps teams can detect infrastructure anomalies — **CPU spikes, RAM overload, network latency bursts** — using Machine Learning before failures actually occur.

The system:
- Generates realistic synthetic server metrics with injected anomaly patterns
- Trains an unsupervised **Isolation Forest** model on normal behaviour only
- Streams metric events in real time and scores each one
- Displays everything on a **live Streamlit dashboard**
- Suggests concrete **remediation steps** via a rule-based AI Agent

---

## 🏗️ Architecture

```
infrastructure_metrics.csv
         ↓
  data_generator.py  →  2,000 time-steps, 3 anomaly patterns injected
         ↓
  train_model.py     →  Isolation Forest (trains on normal data only)
         ↓
  monitor.py         →  streaming generator · score_row() · diagnose_anomaly()
         ↓
  agent.py           →  rule-based remediation recommender
         ↓
  app.py             →  Streamlit dashboard (auto-refreshes every 800ms)
```

---

## 📁 Project Structure

```
predictive-monitoring/
├── data_generator.py    # Synthetic metric dataset with anomaly patterns
├── train_model.py       # Isolation Forest training + evaluation
├── monitor.py           # Real-time streaming engine
├── agent.py             # Rule-based DevOps remediation agent
├── app.py               # Streamlit live dashboard
└── requirements.txt     # Python dependencies
```

---

## 📊 Simulated Anomaly Patterns

| Pattern | Metrics Affected | Description |
|---|---|---|
| **Traffic Spike** | CPU, Request Rate | Sudden burst in compute and traffic |
| **Memory Leak** | RAM | Slow monotonic RAM increase over a window |
| **Latency Burst** | Network Latency | Isolated spike in response time |
| **Normal Baseline** | All | Gaussian noise around healthy levels |

---

## 🤖 ML Model — Isolation Forest

Trained **exclusively on normal rows** (unsupervised). Learns to isolate anomalies by randomly partitioning feature space — anomalous points are easier to isolate and get shorter average path lengths.

| Metric | Value |
|---|---|
| Overall Accuracy | **88%** |
| Anomaly Precision | **95%** |
| Anomaly Recall | **85%** |
| Anomaly F1 Score | **0.90** |
| Contamination | 0.08 |
| n_estimators | 200 |

---

## 📺 Dashboard Features

- 📈 Live **CPU / RAM / Network Latency** charts (rolling 120-sample window)
- 🔴 **Anomaly Score** time-series with alert threshold line
- 🗓️ **Anomaly Timeline** — scatter plot of all flagged events
- ⚠️ **Alert Feed** — last 15 anomaly events with timestamps and flags
- 🤖 **Agent Actions** panel — live remediation suggestions
- 📊 **KPI cards** — CPU, RAM, Latency, Request Rate, Anomaly Score
- 🔮 **BONUS** — Predicted Anomaly banner warns ~2 minutes before an anomaly window

---

## 🛠️ DevOps Agent — Remediation Actions

| Trigger | Suggested Actions |
|---|---|
| CPU > 85% | Scale out instances · `kubectl top pods` · Enable HPA |
| RAM > 90% | Restart containers · Inspect heap dumps · Set memory limits |
| Latency > 300ms | Redistribute traffic · Circuit-breaker · Check DNS/CDN |
| Disk I/O > 90% | Move hot data to Redis · Throttle batch jobs |
| Request Rate > 900 | Rate-limit at API gateway · Auto-scaling warm-up · WAF rules |

---

## 🚀 Quick Start

### 1. Clone & install dependencies
```bash
git clone https://github.com/your-username/predictive-monitoring.git
cd predictive-monitoring
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```
> Generates `infrastructure_metrics.csv` and saves `isolation_forest.pkl` + `scaler.pkl`

### 3. (Optional) Run terminal simulation
```bash
python monitor.py
```

### 4. Launch the dashboard
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 📦 Requirements

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
joblib>=1.3
streamlit>=1.32
plotly>=5.20
streamlit-autorefresh>=1.0.1
```

- Python **3.10+** required

---

## 📌 Features at a Glance

- ✅ Synthetic data generation with realistic patterns
- ✅ Unsupervised anomaly detection (no labels needed at inference)
- ✅ Real-time metric streaming simulation
- ✅ Live interactive dashboard
- ✅ Automated remediation suggestions
- ✅ Predictive anomaly warning (bonus feature)

---

*Built with Python · scikit-learn · Streamlit · Plotly*
