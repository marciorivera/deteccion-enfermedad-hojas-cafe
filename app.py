"""
Detección de Enfermedades en Hojas de Café
Streamlit + TensorFlow (MobileNetV2) + API de Groq
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from groq_service import get_disease_guidance

# --------------------------------------------------------------------------
# Configuración de la página
# --------------------------------------------------------------------------
st.set_page_config(
    page_title=" | Diagnóstico Foliar de Café - Marcio Rivera",
    page_icon="🌿",
    layout="wide",
)

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "coffee_disease_model.keras"
CLASS_INDICES_PATH = APP_DIR / "class_indices.json"
IMG_SIZE = (224, 224)

# Metadata de presentación por clase (nombre visible, nombre científico, color)
DISEASE_INFO = {
    "healthy": {
        "label": "Hoja Sana",
        "scientific": "Sin patógeno detectado",
        "color": "#2f5233",
    },
    "rust": {
        "label": "Roya del Café",
        "scientific": "Hemileia vastatrix",
        "color": "#b5651d",
    },
    "phoma": {
        "label": "Phoma",
        "scientific": "Phoma spp.",
        "color": "#7a3b3b",
    },
    "leaf_miner": {
        "label": "Minador de la Hoja",
        "scientific": "Leucoptera coffeella",
        "color": "#8a6d1f",
    },
    "cercospora": {
        "label": "Cercospora / Mancha de Hierro",
        "scientific": "Cercospora coffeicola",
        "color": "#5c4a1e",
    },
    "red_spider": {
        "label": "Ácaro Rojo",
        "scientific": "Oligonychus spp.",
        "color": "#8b2e2e",
    },
}

# --------------------------------------------------------------------------
# Estilos (inspirados en el mockup: fondo crema, tarjetas verdes, serif títulos)
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #f7f4ee; }
    h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif; color: #1f2a1f; }
    .adg-eyebrow {
        font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
        color: #8a8478; font-weight: 600; margin-bottom: 0.2rem;
    }
    .adg-confidence {
        font-size: 2.6rem; font-weight: 700; color: #1f2a1f; line-height: 1;
    }
    .adg-card {
        background-color: #ffffff; border: 1px solid #e7e2d6; border-radius: 10px;
        padding: 1.1rem 1.3rem; margin-bottom: 0.7rem;
    }
    .adg-step-num {
        display: inline-block; background-color: #2f5233; color: white;
        border-radius: 6px; width: 26px; height: 26px; text-align: center;
        line-height: 26px; font-size: 0.8rem; font-weight: 700; margin-right: 0.6rem;
    }
    .adg-step-title { font-weight: 700; color: #1f2a1f; font-size: 0.95rem; }
    .adg-step-body { color: #4a473d; font-size: 0.88rem; margin-top: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Carga del modelo (cacheado en memoria entre interacciones)
# --------------------------------------------------------------------------
@st.cache_resource
def load_model_and_classes():
    if not MODEL_PATH.exists() or not CLASS_INDICES_PATH.exists():
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # raw: {"0": "healthy", "1": "rust", ...} -> lista ordenada por índice
    class_names = [raw[str(i)] for i in range(len(raw))]
    return model, class_names


def predict(model, class_names, image: Image.Image):
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx]) * 100
    return class_names[idx], confidence, probs


def init_history():
    if "history" not in st.session_state:
        st.session_state.history = []


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
init_history()
model, class_names = load_model_and_classes()

left, right = st.columns([1, 1.15], gap="large")

with left:
    st.markdown("<div class='adg-eyebrow'>Captura de Imagen Foliar</div>", unsafe_allow_html=True)
    st.title("Diagnóstico de Hoja de Café")
    st.write(
        "Posicione la hoja de café bajo luz natural. El sistema detectará "
        "automáticamente signos de Roya, Phoma, Minador"
        + (", Cercospora o Ácaro Rojo" if class_names and len(class_names) > 4 else "")
        + "."
    )

    tab_upload, tab_camera = st.tabs(["📁 Subir archivo", "📷 Usar cámara"])
    uploaded_file = None
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Selecciona una imagen", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )
    with tab_camera:
        camera_file = st.camera_input("Captura una foto", label_visibility="collapsed")
        if camera_file is not None:
            uploaded_file = camera_file

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    analyze_clicked = st.button(
        "🔬 Analizar Imagen", use_container_width=True, disabled=(image is None or model is None)
    )

    if model is None:
        st.warning(
            "No se encontró el modelo entrenado. Coloca `coffee_disease_model.keras` "
            "y `class_indices.json` (generados por el notebook de entrenamiento) "
            "en la carpeta `app/`."
        )

with right:
    if analyze_clicked and image is not None and model is not None:
        with st.spinner("Analizando imagen..."):
            pred_class, confidence, probs = predict(model, class_names, image)

        info = DISEASE_INFO.get(pred_class, {"label": pred_class, "scientific": "", "color": "#2f5233"})

        with st.spinner("Generando recomendaciones técnicas con IA..."):
            guidance = get_disease_guidance(info["label"], info["scientific"], confidence)

        st.session_state.history.insert(
            0,
            {
                "label": info["label"],
                "confidence": confidence,
                "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
            },
        )
        st.session_state.last_result = {
            "info": info,
            "confidence": confidence,
            "guidance": guidance,
        }

    result = st.session_state.get("last_result")

    if result:
        info = result["info"]
        guidance = result["guidance"]

        top_l, top_r = st.columns([2.2, 1])
        with top_l:
            st.markdown("<div class='adg-eyebrow'>Último diagnóstico</div>", unsafe_allow_html=True)
            st.markdown(f"## {info['label']}")
            st.caption(f"*{info['scientific']}* · Detectado recientemente")
        with top_r:
            st.markdown(
                f"<div class='adg-confidence'>{result['confidence']:.1f}%</div>"
                "<div class='adg-eyebrow'>Confianza IA</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("##### 🌱 Orientación y Manejo Preventivo")
        st.caption("Recomendación técnica generada automáticamente para esta situación:")

        for i, step in enumerate(guidance, start=1):
            st.markdown(
                f"""
                <div class="adg-card">
                    <span class="adg-step-num">{i:02d}</span>
                    <span class="adg-step-title">{step['title']}</span>
                    <div class="adg-step-body">{step['body']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.session_state.history:
            st.markdown("---")
            st.markdown("##### 🕘 Historial reciente")
            for h in st.session_state.history[:5]:
                c1, c2 = st.columns([3, 1])
                c1.write(f"🟢 {h['label']}")
                c2.caption(h["timestamp"])
    else:
        st.info("Sube o captura una foto de una hoja de café y presiona **Analizar Imagen** para ver el diagnóstico.")

