"""
monitor.py — Système de surveillance de tension, indépendant de l'interface.
Peut être importé par une CLI, une API Flask, ou un scheduler temps réel.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Levée quand le modèle n'a pas encore été chargé."""


class VoltageMonitor:
    """
    Charge le modèle sauvegardé et expose deux méthodes publiques :
        - add_measurement(voltage)  → alimente l'historique
        - predict_and_evaluate()    → prédit + évalue selon la norme CEI 60038
    """

    LEVEL_INFO = {
        "CRITICAL": {"emoji": "🔴", "relay": "EMERGENCY_STOP"},
        "ALERT"   : {"emoji": "🟠", "relay": "WARNING_OUTPUT"},
        "WARNING" : {"emoji": "🟡", "relay": "NO_ACTION"},
        "NORMAL"  : {"emoji": "🟢", "relay": "NO_ACTION"},
    }

    def __init__(self, model_dir: Path = None):
        self._model_dir   = model_dir or config.MODEL_DIR
        self._model       = None
        self._scaler_X    = None
        self._scaler_y    = None
        self._feature_cols = None
        self._buffer: list[float] = []
        self._min_history = max(config.LAGS)  # au moins 12 mesures

    # ── Chargement ────────────────────────────────────────────────────────────

    def load(self) -> "VoltageMonitor":
        """Charge les artefacts depuis model_dir. Chaînable."""
        needed = ["xgboost_model.pkl", "scaler_X.pkl",
                  "scaler_y.pkl", "feature_cols.pkl"]
        missing = [f for f in needed if not (self._model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Fichiers manquants dans {self._model_dir} : {missing}\n"
                "Lancez d'abord : python train.py"
            )

        self._model        = joblib.load(self._model_dir / "xgboost_model.pkl")
        self._scaler_X     = joblib.load(self._model_dir / "scaler_X.pkl")
        self._scaler_y     = joblib.load(self._model_dir / "scaler_y.pkl")
        self._feature_cols = joblib.load(self._model_dir / "feature_cols.pkl")
        log.info("Modèle chargé depuis %s", self._model_dir.resolve())
        return self

    # ── Mesure ────────────────────────────────────────────────────────────────

    def add_measurement(self, voltage: float) -> int:
        """Ajoute une mesure au buffer. Retourne la taille courante."""
        self._buffer.append(float(voltage))
        if len(self._buffer) > 20:
            self._buffer.pop(0)
        return len(self._buffer)

    # ── Prédiction ────────────────────────────────────────────────────────────

    def predict_next(self) -> Optional[float]:
        """Retourne la tension prédite au prochain pas ou None si buffer trop court."""
        if self._model is None:
            raise ModelNotLoadedError("Appelez .load() avant de prédire.")
        if len(self._buffer) < self._min_history:
            return None

        series = pd.Series(self._buffer)
        hour   = datetime.now().hour

        features: dict = {}
        for lag in config.LAGS:
            features[f"lag_{lag}"] = series.shift(lag).iloc[-1]
        for d in config.DIFF_ORDERS:
            features[f"diff_{d}"] = series.diff(d).iloc[-1]
        for w in config.MA_WINDOWS:
            features[f"ma_{w}"] = series.rolling(w).mean().iloc[-1]

        features["hour"]     = hour
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # Réordonner selon l'ordre d'entraînement
        df_feat   = pd.DataFrame([features])[self._feature_cols]
        X_scaled  = self._scaler_X.transform(df_feat)
        pred_s    = self._model.predict(X_scaled)
        predicted = self._scaler_y.inverse_transform(
            pred_s.reshape(-1, 1)
        )[0, 0]
        return float(predicted)

    # ── Évaluation normative ─────────────────────────────────────────────────

    def evaluate(self, voltage: float) -> dict:
        """Classe une tension selon la norme CEI 60038. Retourne un dict."""
        t = config.THRESHOLDS
        v = voltage

        if v >= t["CRITICAL_HIGH"]:
            level, action = "CRITICAL", "ARRÊT D'URGENCE — surtension sévère"
        elif v >= t["ALERT_HIGH"]:
            level, action = "ALERT",    "Réduction de charge"
        elif v >= t["WARNING_HIGH"]:
            level, action = "WARNING",  "Surveillance — tension haute"
        elif v <= t["CRITICAL_LOW"]:
            level, action = "CRITICAL", "ARRÊT D'URGENCE — sous-tension sévère"
        elif v <= t["ALERT_LOW"]:
            level, action = "ALERT",    "Vérifier alimentation"
        elif v <= t["WARNING_LOW"]:
            level, action = "WARNING",  "Surveillance — tension basse"
        else:
            level, action = "NORMAL",   "Aucune action requise"

        info = self.LEVEL_INFO[level]
        return {
            "level"  : level,
            "action" : action,
            "relay"  : info["relay"],
            "emoji"  : info["emoji"],
        }

    # ── Cycle complet ─────────────────────────────────────────────────────────

    def run_cycle(self, measured_voltage: float) -> dict:
        """
        Pipeline en une seule ligne :
            mesure → prédiction → évaluation normative.
        Retourne un dict avec toutes les informations du cycle.
        """
        self.add_measurement(measured_voltage)
        predicted = self.predict_next()

        if predicted is None:
            return {
                "status"  : "COLLECTING",
                "message" : f"{len(self._buffer)}/{self._min_history} mesures collectées",
            }

        decision = self.evaluate(predicted)
        return {
            "status"       : "OK",
            "measured_v"   : round(measured_voltage, 2),
            "predicted_v"  : round(predicted, 2),
            "level"        : decision["level"],
            "action"       : decision["action"],
            "relay"        : decision["relay"],
            "emoji"        : decision["emoji"],
        }
