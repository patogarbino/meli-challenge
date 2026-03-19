"""Streamlit UI for the MeLi new/used classifier.

Upload a JSONL file → calls FastAPI backend → shows predictions.
"""

import io
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Credentials (for production use env vars or a proper auth provider)
VALID_USERS = {
    "admin": "Meli",
}

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MeLi Classifier",
    page_icon="🛒",
    layout="wide",
)

# ── Auth ───────────────────────────────────────────────────────────────────────

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
    <style>
        .login-header {
            font-size: 2.5rem; font-weight: 700; text-align: center;
            background: linear-gradient(135deg, #FFE600 0%, #3483FA 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="login-header">🛒 MeLi Classifier</p>', unsafe_allow_html=True)
    st.markdown("---")

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.subheader("🔐 Iniciar sesión")
        with st.form("login_form"):
            username = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            submit = st.form_submit_button("Ingresar", use_container_width=True)

        if submit:
            if username in VALID_USERS and VALID_USERS[username] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("❌ Usuario o contraseña incorrectos")

    st.stop()

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FFE600 0%, #3483FA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #666;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .stDataFrame {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">🛒 MeLi New/Used Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Subí un archivo JSONL con items de MercadoLibre y obtené predicciones al instante</p>', unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"👤 **{st.session_state.username}**")
    if st.button("🚪 Cerrar sesión", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()

    st.markdown("---")
    st.header("ℹ️ Acerca de")
    st.markdown("""
    Este clasificador utiliza un modelo **CatBoost** entrenado con 
    arquitectura hexagonal para predecir si un artículo de MercadoLibre 
    es **nuevo** o **usado**.
    
    **Métricas del modelo:**
    - Accuracy: **91.4%**
    - AUC-ROC: **0.9734**
    
    ---
    
    **Formato del archivo:**
    
    El archivo debe ser `.jsonlines` con un item JSON por línea,
    con la estructura de la API de MercadoLibre.
    """)

# ── File upload ────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "📁 Arrastrá o seleccioná un archivo .jsonlines",
    type=["jsonlines", "jsonl", "json"],
    help="Cada línea debe ser un JSON con la estructura de un item de MeLi",
)

if uploaded_file is not None:
    # Only call the API if we have a new file (avoid re-running on filter changes)
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("last_file_key") != file_key:
        with st.spinner("🔄 Procesando predicciones..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/json")}
                response = requests.post(f"{API_URL}/api/predict", files=files, timeout=120)

                if response.status_code != 200:
                    st.error(f"❌ Error del servidor: {response.json().get('detail', 'Unknown error')}")
                    st.stop()

                st.session_state.prediction_data = response.json()
                st.session_state.last_file_key = file_key
            except requests.ConnectionError:
                st.error("❌ No se pudo conectar con el servidor. Verificá que la API esté corriendo.")
                st.stop()
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.stop()

    data = st.session_state.prediction_data

    # ── Summary metrics ────────────────────────────────────────────────────

    st.markdown("---")
    st.subheader("📊 Resumen de predicciones")

    summary = data["summary"]
    total = data["total_items"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Items", f"{total:,}")
    with col2:
        new_count = summary.get("new", 0)
        st.metric("🆕 Nuevos", f"{new_count:,}", f"{new_count/total*100:.1f}%")
    with col3:
        used_count = summary.get("used", 0)
        st.metric("♻️ Usados", f"{used_count:,}", f"{used_count/total*100:.1f}%")

    # ── Predictions table ──────────────────────────────────────────────────

    st.markdown("---")
    st.subheader("📋 Predicciones detalladas")

    df = pd.DataFrame(data["predictions"])

    # Color coding
    def highlight_prediction(val):
        if val == "new":
            return "background-color: #d4edda; color: #155724"
        return "background-color: #f8d7da; color: #721c24"

    # Display controls
    col_filter, col_sort = st.columns(2)
    with col_filter:
        filter_option = st.selectbox(
            "Filtrar por:",
            ["Todos", "Solo nuevos", "Solo usados"],
        )
    with col_sort:
        sort_option = st.selectbox(
            "Ordenar por:",
            ["Índice", "Probabilidad (mayor)", "Probabilidad (menor)"],
        )

    # Apply filters
    df_display = df.copy()
    if filter_option == "Solo nuevos":
        df_display = df_display[df_display["prediction"] == "new"]
    elif filter_option == "Solo usados":
        df_display = df_display[df_display["prediction"] == "used"]

    # Apply sorting
    if sort_option == "Probabilidad (mayor)":
        df_display = df_display.sort_values("prob_new", ascending=False)
    elif sort_option == "Probabilidad (menor)":
        df_display = df_display.sort_values("prob_new", ascending=True)

    st.dataframe(
        df_display.style.applymap(highlight_prediction, subset=["prediction"]),
        use_container_width=True,
        height=500,
    )

    # ── Download button ────────────────────────────────────────────────────

    st.markdown("---")
    csv = df.to_csv(index=False)
    st.download_button(
        label="⬇️ Descargar resultados como CSV",
        data=csv,
        file_name="predicciones_meli.csv",
        mime="text/csv",
    )
else:
    # ── Empty state ────────────────────────────────────────────────────────
    st.info("👆 Subí un archivo JSONL para empezar")

    with st.expander("📝 Ejemplo de formato del archivo"):
        st.code(
            '{"title": "iPhone 15", "price": 500000, "condition": "new", "seller_address": {...}, ...}\n'
            '{"title": "Samsung S21 Usado", "price": 200000, "condition": "used", "seller_address": {...}, ...}',
            language="json",
        )
