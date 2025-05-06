import streamlit as st
from PIL import Image
import torch
from torchvision import models
import torch.nn as nn
import requests
import os
import mlflow
from utils.transform import get_transform
from utils.inference import predict_image, log_prediction_to_mlflow

# -----------------------------
# Configuración del modelo
# -----------------------------
MODEL_URL = "https://huggingface.co/AlexReinoso/trabajoTFM/resolve/main/best_model_vangogh.pth"
MODEL_PATH = "models/best_model_vangogh.pth"
CLASS_NAMES = ["Falsa", "Verdadera"]

# -----------------------------
# Cargar modelo con cache
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 2)
    )

    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with st.spinner("Descargando el modelo..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
        st.success("✅ Modelo descargado correctamente.")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# -----------------------------
# Interfaz Streamlit
# -----------------------------
st.set_page_config(page_title="Autenticador Van Gogh", layout="centered")
st.title("🎨 Autenticador de Obras de Van Gogh")
st.markdown("Sube una imagen para predecir si una obra es **verdadera** o **falsa**.")

uploaded_file = st.file_uploader("📤 Sube tu imagen aquí", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Imagen subida", use_column_width=True)

    model = load_model()
    transform = get_transform()

    st.write("⏳ Clasificando...")
    label, confidence, probs = predict_image(image, model, transform, CLASS_NAMES)

    st.success(f"🎯 Predicción: **{label}** con una confianza de **{confidence * 100:.2f}%**")

    st.subheader("Distribución de Probabilidades:")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"**{class_name}:** {probs[i].item() * 100:.2f}%")

    try:
        log_prediction_to_mlflow(image, label, confidence, probs)
        st.info("✅ Resultado registrado en MLflow.")
    except Exception as e:
        st.warning(f"⚠️ No se pudo registrar en MLflow: {e}")
