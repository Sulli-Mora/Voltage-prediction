# setup.py - Exécute l'entraînement au premier déploiement
import subprocess
import os

if not os.path.exists("models/xgboost_model.pkl"):
    print("⚙️ Entraînement du modèle en cours...")
    subprocess.run(["python", "train.py"])
    print("✅ Modèle entraîné !")
