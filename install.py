"""
One-shot installer for the Delhivery ETA Optimizer project.
Run this ONCE before opening any notebook:

    python install.py

Handles:
  1. Core packages (pandas, numpy, seaborn, plotly, etc.)
  2. Graph packages (networkx, node2vec, python-louvain)
  3. ML packages (xgboost, lightgbm, shap)
  4. PyTorch (CPU version — change to GPU below if you have CUDA)
  5. PyTorch Geometric (torch-geometric, torch-scatter, torch-sparse)
  6. NLP packages (spacy, sentence-transformers, transformers, datasets)
  7. Dashboard (streamlit, folium, streamlit-folium, pyvis)
  8. spaCy model download (en_core_web_sm)
"""

import subprocess
import sys
import platform

PY = sys.executable

def run(cmd: str, label: str = ""):
    print(f"\n{'='*60}")
    print(f"Installing: {label or cmd[:60]}")
    print('='*60)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"  WARNING: Non-zero exit for '{label}'. Check output above.")
    else:
        print(f"  OK: {label}")

# ── Step 1: Core data science ──────────────────────────────────────────────────
run(f'{PY} -m pip install --upgrade pip', 'pip upgrade')
run(f'{PY} -m pip install pandas numpy scipy scikit-learn', 'Core data science')

# ── Step 2: Visualisation ──────────────────────────────────────────────────────
run(f'{PY} -m pip install matplotlib seaborn plotly folium pyvis', 'Visualisation')

# ── Step 3: Graph ──────────────────────────────────────────────────────────────
run(f'{PY} -m pip install networkx node2vec python-louvain', 'Graph tools')

# ── Step 4: ML models ─────────────────────────────────────────────────────────
run(f'{PY} -m pip install xgboost lightgbm shap', 'ML models (XGBoost, LightGBM, SHAP)')

# ── Step 5: PyTorch (CPU) ─────────────────────────────────────────────────────
# If you have an NVIDIA GPU, replace this with the CUDA version from pytorch.org
run(
    f'{PY} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu',
    'PyTorch (CPU build)',
)

# ── Step 6: PyTorch Geometric ─────────────────────────────────────────────────
# Must be installed AFTER PyTorch so it can detect the correct version
run(f'{PY} -m pip install torch-geometric', 'torch-geometric')

# torch-scatter and torch-sparse require the exact PyTorch + CPU/CUDA tag
# This installs the CPU prebuilt wheels for PyTorch 2.x
run(
    f'{PY} -m pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html',
    'torch-scatter + torch-sparse (PyG extras, CPU)',
)

# ── Step 7: NLP ───────────────────────────────────────────────────────────────
run(f'{PY} -m pip install spacy', 'spaCy')
run(f'{PY} -m pip install sentence-transformers', 'sentence-transformers')
run(f'{PY} -m pip install "transformers>=4.35.0" datasets', 'HuggingFace transformers + datasets')
run(f'{PY} -m pip install fuzzywuzzy python-Levenshtein', 'fuzzywuzzy')

# ── Step 8: Dashboard ─────────────────────────────────────────────────────────
run(f'{PY} -m pip install streamlit streamlit-folium', 'Streamlit dashboard')

# ── Step 9: Utilities ─────────────────────────────────────────────────────────
run(f'{PY} -m pip install tqdm joblib PyYAML python-dotenv ipykernel', 'Utilities')

# ── Step 10: spaCy English model ──────────────────────────────────────────────
run(f'{PY} -m spacy download en_core_web_sm', 'spaCy model (en_core_web_sm)')

# ── Verify key imports ─────────────────────────────────────────────────────────
print('\n' + '='*60)
print('VERIFICATION')
print('='*60)
checks = [
    ('pandas', 'import pandas; print(f"  pandas {pandas.__version__}")'),
    ('numpy', 'import numpy; print(f"  numpy {numpy.__version__}")'),
    ('seaborn', 'import seaborn; print(f"  seaborn {seaborn.__version__}")'),
    ('plotly', 'import plotly; print(f"  plotly {plotly.__version__}")'),
    ('networkx', 'import networkx; print(f"  networkx {networkx.__version__}")'),
    ('torch', 'import torch; print(f"  torch {torch.__version__}  CUDA={torch.cuda.is_available()}")'),
    ('torch_geometric', 'import torch_geometric; print(f"  torch_geometric {torch_geometric.__version__}")'),
    ('xgboost', 'import xgboost; print(f"  xgboost {xgboost.__version__}")'),
    ('lightgbm', 'import lightgbm; print(f"  lightgbm {lightgbm.__version__}")'),
    ('sentence_transformers', 'from sentence_transformers import SentenceTransformer; print("  sentence_transformers OK")'),
    ('transformers', 'import transformers; print(f"  transformers {transformers.__version__}")'),
    ('spacy', 'import spacy; nlp=spacy.load("en_core_web_sm"); print(f"  spacy {spacy.__version__} + en_core_web_sm OK")'),
    ('streamlit', 'import streamlit; print(f"  streamlit {streamlit.__version__}")'),
    ('folium', 'import folium; print(f"  folium {folium.__version__}")'),
    ('shap', 'import shap; print(f"  shap {shap.__version__}")'),
]

failed = []
for name, check_code in checks:
    result = subprocess.run([PY, '-c', check_code], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print(f"  FAILED: {name} — {result.stderr.strip()[:80]}")
        failed.append(name)

print('\n' + '='*60)
if failed:
    print(f"FAILED PACKAGES: {', '.join(failed)}")
    print("Re-run install.py or install manually with: pip install <package>")
else:
    print("ALL PACKAGES INSTALLED SUCCESSFULLY")
    print("\nNext steps:")
    print("  1. Place Delhivery CSV in:  data/raw/")
    print("  2. Open notebooks in order: 01_EDA → 10_Model_Comparison")
    print("  3. Launch dashboard:        streamlit run dashboard/app.py")
print('='*60)
