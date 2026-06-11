# Delhivery ETA Optimizer — Multi-City Graph Intelligence System

> **CAC IIT Guwahati Summer Projects '26**  
> Problem: Optimizing Delivery ETAs with Graph-Based Network Intelligence (extended to multi-city)

---

## What This Project Does

Delhivery's OSRM system underestimates actual delivery time on a significant fraction of routes because it treats each trip independently. This system models the entire logistics network as a **directed weighted graph**, uses **Graph Neural Networks + LSTM + Transformer** for ETA prediction, and applies **NLP** for hub name parsing and route embeddings — going well beyond the base problem.

**Key differentiator:** Two-level hierarchical graph (city super-graph + per-city facility subgraphs) covering 15 major Indian cities.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & EDA | pandas, numpy, plotly, seaborn |
| NLP | spaCy, sentence-transformers, HuggingFace BERT |
| Graph | NetworkX, node2vec, python-louvain |
| ML Baseline | XGBoost, LightGBM, SHAP |
| DL Models | PyTorch + PyTorch Geometric (GraphSAGE, GAT, LSTM, Transformer) |
| Dashboard | Streamlit + Folium + PyVis |

---

## Project Structure

```
delhivery-eta-optimizer/
├── data/
│   ├── raw/                    ← Download Delhivery dataset here (Kaggle)
│   ├── processed/              ← Auto-generated cleaned files
│   └── city_augmented/        ← Multi-city synthetic extension
├── src/
│   ├── data/                   ← pipeline.py, feature_eng.py, city_extender.py
│   ├── nlp/                    ← address_parser.py, route_embedder.py, delay_classifier.py
│   ├── graph/                  ← builder.py, analytics.py, node2vec_emb.py, hierarchical.py
│   ├── models/                 ← baseline.py, gnn_models.py, lstm_eta.py, transformer_eta.py, ftl_carting.py
│   └── viz/                   ← network_plot.py, metrics_plot.py
├── notebooks/                  ← 01_EDA through 10_Model_Comparison
├── dashboard/                  ← Streamlit app (app.py + 4 pages)
├── reports/                    ← strategy_memo.md, technical_report.md
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Download dataset

Download the Delhivery dataset from Kaggle and place the CSV in `data/raw/`.

```
Kaggle search: "Delhivery Logistics Dataset" → delhivery_data.csv → data/raw/
```

### 3. Run notebooks in order

```
01_EDA.ipynb          → EDA and data understanding
02_NLP_Pipeline.ipynb → Address parsing + route embeddings
03_Graph_Construction → Build graph + node2vec
04_Bottleneck_Analysis → Centrality + bottleneck ranking
05_Baseline_Models     → XGBoost + LightGBM + SHAP
06_GNN_Models          → GraphSAGE + GAT
07_LSTM_Transformer    → Sequential models
08_FTL_Carting         → Route type framework
09_MultiCity_Analysis  → Hierarchical multi-city graph
10_Model_Comparison    → Final benchmark table
```

### 4. Launch dashboard

```bash
streamlit run dashboard/app.py
```

---

## Model Results (Indicative)

| Model | MAE (min) | Within 15% |
|---|---|---|
| XGBoost (No Graph) | ~48 | ~61% |
| LightGBM + Graph Features | ~38 | ~71% |
| GraphSAGE | ~31 | ~77% |
| GAT | ~30 | ~78% |
| LSTM (Multi-hop) | ~27 | ~80% |
| **Transformer (Temporal)** | **~25** | **~83%** |

Graph-enhanced models outperform the baseline by **~48% MAE reduction**.

---

## Three Novel Contributions

1. **Multi-city hierarchical graph** — two-level architecture (city super-graph + facility subgraphs) covering 15 Indian cities with both real and synthetic inter-city corridors.

2. **NLP-enriched graph edges** — sentence-transformer embeddings of corridor descriptions used as edge features in GraphSAGE/GAT, enabling zero-shot delay estimation for new routes.

3. **BERT delay classifier** — classifies delay root causes (traffic / weather / hub congestion / breakdown / last-mile) enabling targeted, specific interventions rather than generic "reduce delay" recommendations.

---

## Reports

- [Strategy Memo](reports/strategy_memo.md) — for Head of Network Operations
- [Technical Report](reports/technical_report.md) — 8-page technical deep-dive
