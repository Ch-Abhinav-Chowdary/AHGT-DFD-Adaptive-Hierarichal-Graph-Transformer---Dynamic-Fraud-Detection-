"""Central configuration for paths and hyperparameters."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Elliptic Bitcoin Dataset (place CSVs in data/raw/)
ELLIPTIC_FEATURES_FILE = RAW_DATA_DIR / "elliptic_txs_features.csv"
ELLIPTIC_CLASSES_FILE = RAW_DATA_DIR / "elliptic_txs_classes.csv"
ELLIPTIC_EDGELIST_FILE = RAW_DATA_DIR / "elliptic_txs_edgelist.csv"

# Cached artifacts
GRAPH_CACHE_FILE = PROCESSED_DATA_DIR / "graph_data.pt"
BASELINE_CHECKPOINT = MODELS_DIR / "baseline.pt"
GRAPH_TRANSFORMER_CHECKPOINT = MODELS_DIR / "graph_transformer.pt"

# ---------------------------------------------------------------------------
# Elliptic dataset columns
# ---------------------------------------------------------------------------
TX_ID_COL = "txId"
TIME_STEP_COL = "time_step"
CLASS_COL = "class"
UNKNOWN_LABEL = "unknown"

# Binary label mapping (Elliptic dataset convention: '1' = illicit, '2' = licit)
LICIT_LABEL = "2"
ILLICIT_LABEL = "1"

# Temporal split boundaries (Elliptic standard: 49 timesteps total)
TRAIN_TIME_STEPS = range(1, 35)   # 1–34
VAL_TIME_STEPS = range(35, 42)    # 35–41
TEST_TIME_STEPS = range(42, 50)   # 42–49

# ---------------------------------------------------------------------------
# Model hyperparameters (used in later stages)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 200                 # 100 → 200: more runway for focal-loss convergence
EARLY_STOPPING_PATIENCE = 20    # 10 → 20: allow longer plateau before stopping

# Baseline GraphSAGE
BASELINE_BATCH_SIZE = 1024
BASELINE_NUM_NEIGHBORS = [25, 10]

# Graph Transformer (Kaggle / T4)
GT_BATCH_SIZE = 512
GT_NUM_NEIGHBORS = [25, 10]
GT_NUM_HEADS = 4
GT_NUM_TRANSFORMER_LAYERS = 3

# Explainability
EXPLAIN_SAMPLE_SIZE = 50
# 200 epochs (was 60): GNNExplainer needs longer optimisation to converge to
# meaningful edge masks, especially with explanation_type='phenomenon'.
EXPLAIN_MAX_EPOCHS = 200

# Visualization
MAX_VIS_NODES = 200
DEFAULT_EGO_HOPS = 2
