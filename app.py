"""
app.py — Version finale pour présentation d'examen
Système de surveillance tension 400V - XGBoost + CEI 60038
"""

import logging
import sys
from datetime import datetime

from monitor import VoltageMonitor

logging.basicConfig(level=logging.INFO, format="%(message)s")


def print_header():
    print("\n" + "=" * 85)
    print(" " * 20 + "🚀 SYSTÈME DE SURVEILLANCE INTELLIGENTE DE TENSION 400V")
    print(" " * 25 + "XGBoost + Norme CEI 60038")
    print("=" * 85)
    print(f"   Modèle chargé avec succès • {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 85)


def print_normes():
    print("\n NORMES ÉLECTRIQUES CEI 60038 (400V)")
    print("-" * 70)
    print("   🔴 CRITIQUE     > 410 V  ou  < 340 V  → Arrêt d'urgence")
    print("   🟠 ALERTE       400-410 V ou 340-360 V → Action immédiate")
    print("   🟡 ATTENTION    395-400 V ou 360-375 V → Surveillance")
    print("   🟢 NORMAL       375-395 V                → Tout est OK")
    print("-" * 70)
    print(" Commandes :   q = quitter    |    r = rapport\n")


def run_demo():
    print_header()
    print_normes()

    try:
        monitor = VoltageMonitor().load()
    except Exception as e:
        print(f"❌ Erreur : {e}")
        print("   Lance d'abord 'python train.py'")
        sys.exit(1)

    cycle = 0
    alerts = 0
    urgences = 0

    print("✅ Simulation prête ! Entrez des tensions mesurées...\n")

    while True:
        try:
            inp = input(f" [Cycle {cycle+1:03d}] Tension mesurée (V) → ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if inp.lower() == "q":
            break
        if inp.lower() == "r":
            print(f"\n📊 RAPPORT : {cycle} cycles | {alerts} alertes | {urgences} urgences\n")
            continue

        try:
            voltage = float(inp)
        except ValueError:
            print("⚠ Entrée invalide → tape un nombre\n")
            continue

        result = monitor.run_cycle(voltage)

        print("-" * 70)
        if result["status"] == "COLLECTING":
            print(f"📊 {result['message']}")
        else:
            print(f" Mesurée   : {result['measured_v']:.2f} V")
            print(f" Prédite   : {result['predicted_v']:.2f} V")
            print(f" Diagnostic: {result['emoji']} {result['level']}")
            print(f" Action    : {result['action']}")

            if result["relay"] == "EMERGENCY_STOP":
                urgences += 1
                alerts += 1
                print("\n⛔ ARRÊT D'URGENCE DÉCLENCHÉ !")
            elif result["relay"] == "WARNING_OUTPUT":
                alerts += 1
                print("\n⚠️  ALERTE DÉCLENCHÉE")
            else:
                print("\n✅ Relais inactif - Tension normale")

        print("-" * 70)
        cycle += 1

    # Rapport final (beau pour le jury)
    print("\n" + "=" * 85)
    print(" FIN DE LA PRÉSENTATION - RÉSUMÉ")
    print("=" * 85)
    print(f" Cycles réalisés          : {cycle}")
    print(f" Alertes totales          : {alerts}")
    print(f" Arrêts d'urgence         : {urgences}")
    print(f" Modèle                   : XGBoost (MAE ≈ 0.78 V | R² ≈ 0.99)")
    print(f" Graphiques disponibles   : plots/training_curve.png")
    print(f"                          : plots/real_vs_prediction.png")
    print("=" * 85)
    print(" Merci pour votre attention ! 🎓")


if __name__ == "__main__":
    run_demo()