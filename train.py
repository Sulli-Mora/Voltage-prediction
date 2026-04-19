"""
train.py — Entraînement, évaluation et sauvegarde du modèle XGBoost.

Usage :
    python train.py
    python train.py --data /chemin/vers/fichier.xlsx
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

import config

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
np.random.seed(42)


# ─── 1. Chargement & validation ───────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            f"Placez votre fichier Excel dans : {path.parent.resolve()}"
        )

    log.info("Chargement de %s ...", path.name)
    df = pd.read_excel(path, engine="openpyxl")

    # Vérification des colonnes obligatoires
    for col in [config.DATETIME_COL, config.TARGET_COL]:
        if col not in df.columns:
            raise ValueError(
                f"Colonne '{col}' absente du fichier. "
                f"Colonnes disponibles : {list(df.columns)}"
            )

    df[config.DATETIME_COL] = pd.to_datetime(df[config.DATETIME_COL])
    df = df.sort_values(config.DATETIME_COL).reset_index(drop=True)

    n = len(df)
    log.info(
        "Dataset : %d lignes | %s → %s | moyenne=%.2f V | std=%.2f V",
        n,
        df[config.DATETIME_COL].min(),
        df[config.DATETIME_COL].max(),
        df[config.TARGET_COL].mean(),
        df[config.TARGET_COL].std(),
    )
    return df


# ─── 2. Feature engineering ───────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Ingénierie des features ...")

    for lag in config.LAGS:
        df[f"lag_{lag}"] = df[config.TARGET_COL].shift(lag)

    for d in config.DIFF_ORDERS:
        df[f"diff_{d}"] = df[config.TARGET_COL].diff(d)

    for w in config.MA_WINDOWS:
        df[f"ma_{w}"] = df[config.TARGET_COL].rolling(w).mean()

    hour = df[config.DATETIME_COL].dt.hour
    df["hour"]     = hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        c for c in df.columns
        if c not in [config.DATETIME_COL, config.TARGET_COL]
    ]
    log.info(
        "%d features | ratio obs/features = %.1f",
        len(feature_cols),
        len(df) / len(feature_cols),
    )
    return df, feature_cols


# ─── 3. Split & normalisation ─────────────────────────────────────────────────

def split_and_scale(df: pd.DataFrame, feature_cols: list):
    n = len(df)
    n_train = int(config.TRAIN_RATIO * n)
    n_val   = int(config.VAL_RATIO * n)

    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train : n_train + n_val]
    test_df  = df.iloc[n_train + n_val :]

    log.info(
        "Split : train=%d | val=%d | test=%d",
        len(train_df), len(val_df), len(test_df)
    )

    scaler_X = RobustScaler()
    scaler_y = RobustScaler()

    X_train = scaler_X.fit_transform(train_df[feature_cols])
    y_train = scaler_y.fit_transform(train_df[[config.TARGET_COL]]).ravel()

    X_val = scaler_X.transform(val_df[feature_cols])
    y_val = scaler_y.transform(val_df[[config.TARGET_COL]]).ravel()

    X_test    = scaler_X.transform(test_df[feature_cols])
    y_test    = test_df[config.TARGET_COL].values
    test_dt   = test_df[config.DATETIME_COL].values

    return (X_train, y_train, X_val, y_val, X_test, y_test, test_dt,
            scaler_X, scaler_y)


# ─── 4. Recherche d'hyperparamètres ──────────────────────────────────────────

def tune_hyperparameters(X_train, y_train):
    log.info("Recherche des hyperparamètres ...")
    tscv       = TimeSeriesSplit(n_splits=config.CV_SPLITS)
    best_score = float("inf")
    best_params = config.BASE_PARAMS.copy()

    grid   = config.PARAM_GRID
    combos = (
        len(grid["max_depth"])
        * len(grid["learning_rate"])
        * len(grid["min_child_weight"])
    )

    with tqdm(total=combos, desc="Grid search", unit="combo") as pbar:
        for depth in grid["max_depth"]:
            for lr in grid["learning_rate"]:
                for mcw in grid["min_child_weight"]:
                    params = {**config.BASE_PARAMS,
                              "max_depth": depth,
                              "learning_rate": lr,
                              "min_child_weight": mcw}

                    scores = []
                    for tr_idx, cv_idx in tscv.split(X_train):
                        m = xgb.XGBRegressor(**params)
                        m.fit(
                            X_train[tr_idx], y_train[tr_idx],
                            eval_set=[(X_train[cv_idx], y_train[cv_idx])],
                            verbose=False,
                        )
                        scores.append(m.best_score)

                    mean = np.mean(scores)
                    if mean < best_score:
                        best_score  = mean
                        best_params = params.copy()
                        pbar.set_postfix(best_mae=f"{best_score:.5f}")
                    pbar.update(1)

    log.info(
        "Meilleur combo : depth=%d | lr=%.3f | min_child=%d | MAE=%.5f",
        best_params["max_depth"],
        best_params["learning_rate"],
        best_params["min_child_weight"],
        best_score,
    )
    return best_params


# ─── 5. Entraînement final ────────────────────────────────────────────────────

def train_final_model(best_params, X_train, y_train, X_val, y_val):
    log.info("Entraînement du modèle final ...")
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    history = model.evals_result().get("validation_0", {}).get("mae", [])
    if history:
        log.info("Meilleure MAE validation : %.5f", min(history))
    return model, history


# ─── 6. Évaluation ───────────────────────────────────────────────────────────

def evaluate(model, scaler_y, X_test, y_test, lag1_test):
    y_pred_s = model.predict(X_test)
    y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mae_pct = mae / np.mean(y_test) * 100

    baseline_mae = mean_absolute_error(y_test, lag1_test)
    improvement  = (baseline_mae - mae) / baseline_mae * 100

    log.info("━━━━━━ Résultats ━━━━━━")
    log.info("MAE  : %.4f V (%.2f%%)", mae, mae_pct)
    log.info("RMSE : %.4f V", rmse)
    log.info("R²   : %.4f", r2)
    log.info("Amélioration vs baseline naïf : +%.1f%%", improvement)

    return y_pred, {"mae": mae, "rmse": rmse, "r2": r2,
                    "mae_pct": mae_pct, "improvement": improvement}


# ─── 7. Graphiques ───────────────────────────────────────────────────────────

def save_plots(val_mae_history, test_dt, y_test, y_pred, metrics):
    config.PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Courbe d'apprentissage ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(1, len(val_mae_history) + 1), val_mae_history,
            linewidth=2, color="#2E86AB", label="MAE validation")
    best = min(val_mae_history)
    ax.axhline(best, color="r", linestyle="--",
               label=f"Meilleure MAE : {best:.5f}")
    ax.set(xlabel="Epoch", ylabel="MAE normalisée",
           title="Courbe d'apprentissage")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p1 = config.PLOT_DIR / "training_curve.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Sauvegardé : %s", p1)

    # --- Réel vs Prédit (6 sous-graphiques) ---
    fig2, axes = plt.subplots(3, 2, figsize=(16, 12))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat
    abs_err = np.abs(y_test - y_pred)
    residuals = y_test - y_pred

    # Série complète
    ax1.plot(test_dt, y_test, label="Réel",  lw=1.5, color="#2E86AB", alpha=0.8)
    ax1.plot(test_dt, y_pred, label="Prédit", lw=1.5, color="#F18F01", alpha=0.8)
    ax1.set(xlabel="Temps", ylabel="Tension (V)",
            title="Réel vs Prédit — période de test")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Zoom 50 premiers
    z = min(50, len(test_dt))
    ax2.plot(test_dt[:z], y_test[:z], label="Réel",  lw=2, color="#2E86AB",
             marker="o", markersize=3)
    ax2.plot(test_dt[:z], y_pred[:z], label="Prédit", lw=2, color="#F18F01",
             marker="x", markersize=3)
    ax2.set(xlabel="Temps", ylabel="Tension (V)",
            title=f"Zoom : {z} premières prédictions")
    ax2.legend(); ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Scatter
    ax3.scatter(y_test, y_pred, alpha=0.5, s=20, color="#A23B72")
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             "r--", lw=2, label="Prédiction parfaite")
    ax3.set(xlabel="Réel (V)", ylabel="Prédit (V)",
            title=f"Corrélation (R²={metrics['r2']:.4f})")
    ax3.legend(); ax3.grid(alpha=0.3)

    # Erreur absolue
    ax4.plot(test_dt, abs_err, lw=1.5, color="#F18F01")
    ax4.axhline(metrics["mae"], color="r", linestyle="--",
                label=f"MAE = {metrics['mae']:.2f} V")
    ax4.fill_between(test_dt, 0, abs_err, alpha=0.3, color="#F18F01")
    ax4.set(xlabel="Temps", ylabel="Erreur absolue (V)",
            title="Erreur de prédiction")
    ax4.legend(); ax4.grid(alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Distribution
    ax5.hist(abs_err, bins=30, edgecolor="black", color="#A23B72", alpha=0.7)
    ax5.axvline(metrics["mae"], color="r", linestyle="--",
                lw=2, label=f"MAE = {metrics['mae']:.2f} V")
    p95 = np.percentile(abs_err, 95)
    ax5.axvline(p95, color="orange", linestyle="--",
                lw=2, label=f"95e centile = {p95:.2f} V")
    ax5.set(xlabel="Erreur absolue (V)", ylabel="Fréquence",
            title="Distribution des erreurs")
    ax5.legend(); ax5.grid(alpha=0.3)

    # Résidus
    ax6.scatter(y_pred, residuals, alpha=0.5, s=20, color="#A23B72")
    ax6.axhline(0, color="r", linestyle="--", lw=2)
    ax6.set(xlabel="Prédit (V)", ylabel="Résidus (V)",
            title="Résidus vs Prédictions")
    ax6.grid(alpha=0.3)

    plt.tight_layout()
    p2 = config.PLOT_DIR / "real_vs_prediction.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    log.info("Sauvegardé : %s", p2)


# ─── 8. Sauvegarde du modèle ─────────────────────────────────────────────────

def save_model(model, scaler_X, scaler_y, feature_cols):
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,     config.MODEL_DIR / "xgboost_model.pkl")
    joblib.dump(scaler_X,  config.MODEL_DIR / "scaler_X.pkl")
    joblib.dump(scaler_y,  config.MODEL_DIR / "scaler_y.pkl")
    joblib.dump(feature_cols, config.MODEL_DIR / "feature_cols.pkl")
    log.info("Modèle sauvegardé dans : %s", config.MODEL_DIR.resolve())


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main(data_path: Path = None):
    data_path = data_path or config.DATA_PATH

    df = load_data(data_path)
    df, feature_cols = build_features(df)

    (X_train, y_train, X_val, y_val,
     X_test, y_test, test_dt,
     scaler_X, scaler_y) = split_and_scale(df, feature_cols)

    best_params    = tune_hyperparameters(X_train, y_train)
    model, history = train_final_model(best_params, X_train, y_train, X_val, y_val)

    # Valeur lag_1 pour le baseline
    lag1_test = df.iloc[
        int(config.TRAIN_RATIO * len(df)) + int(config.VAL_RATIO * len(df)):
    ]["lag_1"].values

    y_pred, metrics = evaluate(model, scaler_y, X_test, y_test, lag1_test)

    if history:
        save_plots(history, test_dt, y_test, y_pred, metrics)

    save_model(model, scaler_X, scaler_y, feature_cols)
    log.info("Entraînement terminé avec succès.")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner le modèle XGBoost tension")
    parser.add_argument("--data", type=Path, default=None,
                        help="Chemin vers le fichier Excel (optionnel)")
    args = parser.parse_args()

    try:
        main(data_path=args.data)
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)
    except Exception as e:
        log.exception("Erreur inattendue : %s", e)
        sys.exit(1)
