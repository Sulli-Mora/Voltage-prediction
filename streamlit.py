import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from monitor import VoltageMonitor

st.set_page_config(page_title="Surveillance Tension 400V", page_icon="⚡", layout="wide")

# Chargement du modèle
@st.cache_resource
def load_monitor():
    return VoltageMonitor().load()

monitor = load_monitor()

# ==================== SIDEBAR ====================
st.sidebar.image("https://img.icons8.com/fluency/96/000000/electricity.png", width=80)
st.sidebar.title("Système Intelligent")
st.sidebar.markdown("**XGBoost + Norme CEI 60038**")
st.sidebar.markdown(f"Modèle chargé le : {datetime.now().strftime('%d/%m/%Y %H:%M')}")

st.sidebar.markdown("### Normes CEI 60038")
st.sidebar.markdown("🔴 **CRITIQUE** → >410V ou <340V → Arrêt d’urgence")
st.sidebar.markdown("🟠 **ALERTE** → 400-410V ou 340-360V")
st.sidebar.markdown("🟡 **ATTENTION** → 395-400V ou 360-375V")
st.sidebar.markdown("🟢 **NORMAL** → 375-395V")

# ==================== MAIN ====================
st.title("🚀 Surveillance Intelligente de Tension 400V")
st.markdown("**Prédiction XGBoost + Analyse selon norme électrique**")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Tension mesurée")
    voltage = st.number_input("Entrez la tension mesurée (V)", 
                              min_value=0.0, max_value=1000.0, 
                              value=380.0, step=0.1)

    if st.button("🔄 Lancer la prédiction", type="primary"):
        result = monitor.run_cycle(voltage)

        if result["status"] == "COLLECTING":
            st.info(result["message"])
        else:
            st.success("Prédiction terminée")

            # Affichage résultat
            st.metric("Tension mesurée", f"{result['measured_v']:.2f} V")
            st.metric("Tension prédite (prochaine)", f"{result['predicted_v']:.2f} V")

            # Diagnostic avec couleur
            color = {"CRITICAL": "🔴", "ALERT": "🟠", "WARNING": "🟡", "NORMAL": "🟢"}
            st.markdown(f"### Diagnostic : {color.get(result['level'], '🟢')} **{result['level']}**")
            st.write(f"**Action :** {result['action']}")

            if result["relay"] == "EMERGENCY_STOP":
                st.error("⛔ ARRÊT D’URGENCE DÉCLENCHÉ !")
            elif result["relay"] == "WARNING_OUTPUT":
                st.warning("⚠️ ALERTE DÉCLENCHÉE")

with col2:
    st.subheader("Graphiques d’entraînement")
    try:
        st.image("plots/real_vs_prediction.png", caption="Réel vs Prédiction")
        st.image("plots/training_curve.png", caption="Courbe d’apprentissage")
    except:
        st.info("Graphiques non trouvés (lance train.py si besoin)")

# Historique des mesures
if st.checkbox("Afficher historique des mesures"):
    st.write("Dernières mesures :", monitor._buffer[-10:])

st.caption("Projet Data Mining - Sulli © 2026")