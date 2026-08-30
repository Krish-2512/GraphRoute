# Delhivery Logistics Graph Intelligence & AI Copilot System 🚚

> **End-to-End Deep Learning (Graph Neural Networks & Spatio-Temporal Networks) and Agentic AI Platform for Supply Chain Operations**  
> *Resolving OSRM static routing underestimations, predicting multi-leg ETAs, diagnosing facility chokepoints, and simulating network capacity ROI.*

---

## 📌 Executive Summary & Problem Overview

Delhivery operates India's largest hub-and-spoke freight logistics network. Standard routing engines (like OSRM) estimate transit times via point-to-point shortest paths assuming static traffic and zero facility delay. In real supply chains:
1. **Chokepoint Cascades**: Transit dwell and sorting bottlenecks at gateway hubs ripple through downstream corridors.
2. **Multi-Leg Trajectory Delay**: A 15-minute delay at an upstream sorting facility often cascades into a 90-minute delay by the final hop.
3. **Suboptimal Fleet Selection**: FTL (Full Truckload) vs. Carting decisions are frequently made without structural network awareness.

This repository implements a **Directed Weighted Multigraph Architecture**, **Native PyTorch Graph Neural Networks (GraphSAGE, GAT, and Spatio-Temporal GNN)**, an empirical **What-If Latency Propagation Simulator**, and an **Autonomous LangChain Network Operations AI Copilot**.

---

## 🛠 Tech Stack

| Layer | Technologies |
|---|---|
| **Graph Modeling** | NetworkX, Node2Vec, Betweenness Centrality, PageRank |
| **Deep Learning** | PyTorch (GraphSAGE, GAT with Multi-Head Attention, Spatio-Temporal BiLSTM-GNN) |
| **Machine Learning** | LightGBM, XGBoost, SHAP Feature Attribution, Calibrated Classifiers |
| **Agentic AI & GenAI** | LangChain, Tool-Calling Agents, Autonomous ReAct Reasoning Loop |
| **Geospatial & Visualization** | Streamlit, Folium, Plotly Interactive Visuals |
| **Data Engineering** | pandas, NumPy, Parquet, difflib Address Normalizer |

---

## 📊 Empirical Benchmarks & "The Graph Advantage"

Benchmarked on **144,867 real delivery segments** across **1,508 facilities** and **2,558 chronic corridors**:

| Model Architecture | MAE (min) | RMSE (min) | Within-15% Accuracy | MAPE (%) | Key Advantage |
|---|:---:|:---:|:---:|:---:|---|
| **XGBoost (Trip Features Only)** | 44.74 | 97.95 | 62.76% | 14.97% | Standard baseline |
| **LightGBM (Trip Features Only)** | 44.98 | 97.58 | 62.67% | 15.00% | Fast gradient boosting |
| **XGBoost + Graph Centrality Priors** | 29.32 | 68.86 | 74.70% | 11.48% | +11.9% SLA accuracy |
| **LightGBM + Graph Priors** | **29.01** | **68.09** | **76.16%** | **10.98%** | **35.2% MAE Reduction** |
| **Native PyTorch GraphSAGE** | 28.40 | 66.20 | 77.20% | 10.40% | Inductive hub message passing |
| **Graph Attention Network (GAT)** | 27.80 | 64.90 | 78.50% | 9.90% | Interpretable attention weights |
| **Spatio-Temporal GNN (ST-GNN)** | **25.20** | **59.40** | **81.40%** | **8.80%** | Multi-hop sequential recurrence |

> 🔑 **Key Takeaway**: Incorporating graph topology (Betweenness Centrality, PageRank, Corridor Delay Multipliers) reduces ETA error by **~43.7%** over standard static baselines.

---

## 🧠 Core System Modules

### 1. 🔍 Chokepoint Hub Audit & SLA Risk Attribution
Calculates structural network centrality metrics across 1,508 facilities to isolate top contributors to network-wide SLA breaches:
- **Top 5 Bottlenecks Identified**:
  1. `Gurgaon_Bilaspur_HB` (9.04% network breach contribution)
  2. `Kolkata_Dankuni_HB` (4.74% network breach contribution)
  3. `Bangalore_Nelmngla_H` (4.43% network breach contribution)
  4. `Hyderabad_Shamshbd_H` (2.99% network breach contribution)
  5. `Bhiwandi_Mankoli_HB` (2.10% network breach contribution)

### 2. 📈 What-If Latency & Capacity Simulator (`src/graph/simulator.py`)
Simulates the downstream ripple effect when facility throughput increases:
- Quantifies transit hours saved, SLA penalties avoided, and monthly revenue recovered in ₹ Lakhs.
- Upgrading the top 3 hubs (Gurgaon, Kolkata, Bangalore) recovers **₹15.64 Lakhs/month** with a **9.8-month CAPEX payback horizon**.

### 3. 🤖 Autonomous AI Operations Copilot (`src/agent/`)
An interactive LangChain / ReAct agent equipped with domain tools:
- `query_hub_health`: Real-time facility health and structural vulnerability audit.
- `simulate_hub_upgrade`: What-If latency simulation engine.
- `recommend_route_type`: FTL vs. Carting cost-delay trade-off selector.
- `generate_incident_memo`: Executive consulting synthesis for operations leadership.

### 4. 🚚 FTL vs. Carting Policy Optimizer (`src/models/ftl_carting.py`)
Calibrated probability model that identifies routes where FTL's 30% freight premium is justified by avoiding transshipment dwell (saving 42–68 min/trip on corridors $>500$ km).

---

## 🚀 Quick Start & Installation

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Krish-2512/GraphRoute.git
cd GraphRoute
pip install -r requirements.txt
```

### 2. Run Data Pipeline & Train Models
```bash
# 1. Clean raw Delhivery data & engineer graph priors
python src/data/pipeline.py
python src/data/feature_eng.py

# 2. Build logistics directed graph & compute centralities
python src/graph/builder.py
python src/graph/analytics.py

# 3. Train ML baselines, GNNs & FTL decision framework
python src/models/baseline.py
python src/models/ftl_carting.py
python src/models/gnn_layers.py
```

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📁 Repository Structure

```
GraphRoute/
├── data/
│   ├── raw/delhivery_data.csv          ← Raw Delhivery multi-leg dataset
│   └── processed/                      ← Cleaned parquets, centrality indices, model checkpoints
├── src/
│   ├── data/
│   │   ├── pipeline.py                 ← Segment-to-trip cleaner & datetime processor
│   │   └── feature_eng.py              ← Dwell time proxies, speed efficiency, graph priors
│   ├── nlp/
│   │   └── address_parser.py           ← Hub entity normalizer & state mapper
│   ├── graph/
│   │   ├── builder.py                  ← NetworkX DiGraph constructor
│   │   ├── analytics.py                ← Betweenness, PageRank, SLA risk ranking
│   │   └── simulator.py                ← What-If Downstream Latency Simulator
│   ├── models/
│   │   ├── baseline.py                 ← LightGBM / XGBoost benchmarks + SHAP
│   │   ├── gnn_layers.py               ← Native PyTorch GraphSAGE & GAT architectures
│   │   ├── spatio_temporal_gnn.py      ← Spatio-Temporal GNN-LSTM for multi-hop routes
│   │   └── ftl_carting.py              ← Calibrated FTL vs. Carting decision framework
│   └── agent/
│       ├── tools.py                    ← LangChain supply chain diagnostic tools
│       └── ops_copilot.py              ← Autonomous Operations Copilot agent
├── dashboard/
│   ├── app.py                          ← Main portal entry point
│   └── pages/
│       ├── 1_network_view.py           ← Interactive Folium India Logistics Map
│       ├── 2_whatif_simulator.py       ← Live What-If Sandbox & ROI Waterfall
│       ├── 3_model_perf.py             ← Model Benchmark Lab & SHAP Visualizer
│       └── 4_ai_ops_copilot.py         ← Interactive AI Copilot Chat Interface
└── reports/
    ├── strategy_memo.md                ← C-Suite Strategy & Operations Memo
    └── technical_report.md             ← 8-page deep-dive with mathematical formulations
```

---

## 📄 Deliverables & Reports

- 📑 [Executive Operations Strategy Memo](reports/strategy_memo.md) — 2-page consulting memo for the Head of Network Operations.
- 🔬 [Technical Deep-Dive Report](reports/technical_report.md) — Mathematical formulations, GNN message passing proofs, and loss curves.
