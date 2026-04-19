"""
config.py — Configuration centralisée du prédicteur de tension.
Modifiez uniquement ce fichier pour adapter le projet à votre environnement.
"""

from pathlib import Path

# ─── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR / "data" / "Data Voltage 400V.xlsx"
MODEL_DIR  = BASE_DIR / "models"
PLOT_DIR   = BASE_DIR / "plots"

# ─── Colonne cible ────────────────────────────────────────────────────────────
TARGET_COL   = "voltage_AB"
DATETIME_COL = "datetime"

# ─── Feature engineering ──────────────────────────────────────────────────────
LAGS         = [1, 2, 3, 6, 12]
DIFF_ORDERS  = [1, 2, 3]
MA_WINDOWS   = [3, 6]

# ─── Split temporel ───────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test_ratio = 1 - TRAIN_RATIO - VAL_RATIO = 0.15

# ─── Hyperparamètres XGBoost (recherche en grille) ────────────────────────────
BASE_PARAMS = {
    "n_estimators"      : 300,
    "learning_rate"     : 0.05,
    "max_depth"         : 4,
    "min_child_weight"  : 3,
    "subsample"         : 0.8,
    "colsample_bytree"  : 0.8,
    "reg_alpha"         : 0.1,
    "reg_lambda"        : 1.0,
    "random_state"      : 42,
    "early_stopping_rounds": 30,
    "eval_metric"       : "mae",
    "verbosity"         : 0,
}

PARAM_GRID = {
    "max_depth"       : [3, 4, 5],
    "learning_rate"   : [0.03, 0.05, 0.07],
    "min_child_weight": [2, 3, 4],
}

CV_SPLITS = 3   # TimeSeriesSplit

# ─── Norme électrique CEI 60038 (tension 400 V) ───────────────────────────────
NOMINAL_VOLTAGE = 400  # V

THRESHOLDS = {
    "CRITICAL_HIGH": 410,   # > 410 V → arrêt d'urgence
    "ALERT_HIGH"   : 400,   # 400-410 V → alerte
    "WARNING_HIGH" : 395,   # 395-400 V → attention
    "WARNING_LOW"  : 375,   # 375-395 V → normal
    "ALERT_LOW"    : 360,   # 360-375 V → attention
    "CRITICAL_LOW" : 340,   # < 340 V  → arrêt d'urgence
}
