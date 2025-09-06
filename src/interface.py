# file: src/interface.py
import os
import binascii
import torch
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
import torch.nn as nn

from utils.transform import get_transform
from utils.inference import predict_image, log_prediction_to_mlflow

APP_VERSION = "interface v3.1"
st.set_page_config(page_title="Autenticador Van Gogh", layout="centered")
st.sidebar.info(APP_VERSION)

# -----------------------------
# Model configuration
# -----------------------------
MODEL_CONFIGS = {
    "ResNet50": {
        "path": "src/models/resnet50_vangogh_2.pth",
        "dropout": 0.5,
        "arch": "resnet50",
    },
    "EfficientNetB0": {
        "path": "src/models/efficientnetb0_vangogh.pth",
        "dropout": 0.5,
        "arch": "efficientnet_b0",
    },
}
CLASS_NAMES = ["Falsa", "Verdadera"]

# Optional: quick debug of model file signature (remove later)
debug_path = MODEL_CONFIGS["ResNet50"]["path"]
if os.path.exists(debug_path):
    with open(debug_path, "rb") as f:
        sig = f.read(16)
    st.sidebar.code(f"{debug_path} size={os.path.getsize(debug_path)} bytes, sig={binascii.hexlify(sig)}")
else:
    st.sidebar.warning(f"Missing: {debug_path}")

# -----------------------------
# Builders and helpers
# -----------------------------
def build_model(arch: str, dropout: float) -> nn.Module:
    if arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, 2))
        return m
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_features = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 2))
        return m
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

def strip_module_prefix(sd: dict) -> dict:
    if sd and next(iter(sd)).startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def sniff_signature(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read(16)

# -----------------------------
# Cached loader
# -----------------------------
@st.cache_resource
def load_model(model_name: str) -> nn.Module:
    cfg = MODEL_CONFIGS[model_name]
    path = cfg["path"]

    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        st.stop()

    size_b = os.path.getsize(path)
    if size_b < 10_000:
        st.error(f"Model file too small ({size_b} bytes). Probably not a valid checkpoint.")
        st.stop()

    sig = sniff_signature(path)
    valid_prefixes = (b"\x80\x04", b"\x80\x05", b"PK\x03\x04")  # pickle or zip
    textual_prefixes = (b"<!DOCTYP", b"<html", b"Error", b"ERROR", b"Export", b"EXPORT", b"{", b"[")
    if not sig.startswith(valid_prefixes):
        if any(sig.startswith(tp) for tp in textual_prefixes):
            st.error(
                "Model file looks like text/HTML or an error message, not a binary .pth.\n"
                "Download the binary file (Raw/Download) or re-export with torch.save(model.state_dict(), ...)."
            )
            st.stop()
        else:
            st.warning(f"Unrecognized checkpoint signature: {binascii.hexlify(sig)}. Trying to load anyway.")

    # build arch
    try:
        model = build_model(cfg["arch"], cfg["dropout"])
    except Exception as e:
        st.error(f"Could not build model: {e}")
        st.stop()

    # load checkpoint
    def _safe_load(p: str):
        try:
            return torch.load(p, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(p, map_location="cpu")

    try:
        ckpt = _safe_load(path)
    except Exception as e:
        st.error(
            "Failed to deserialize the model file.\n"
            "Typical causes: downloaded HTML viewer, corrupted file, or non-PyTorch pickle.\n"
            f"Detail: {e}"
        )
        st.stop()

    # normalize state_dict
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    elif hasattr(ckpt, "state_dict"):
        ckpt = ckpt.state_dict()
    else:
        st.error("Loaded object is not a state_dict or model instance.")
        st.stop()

    ckpt = strip_module_prefix(ckpt)

    # apply weights
    try:
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing or unexpected:
            st.warning(
                "Checkpoint keys do not perfectly match.\n"
                f"Missing: {list(missing)[:5]}{'...' if len(missing)>5 else ''}\n"
                f"Unexpected: {list(unexpected)[:5]}{'...' if len(unexpected)>5 else ''}\n"
                "If this is only the final classifier, it is usually fine."
            )
    except RuntimeError as e:
        st.error("Failed to load weights into the model. Ensure same arch and num classes.\n" + str(e))
        st.stop()

    model.eval()
    return model

# -----------------------------
# UI
# -----------------------------
st.title("🎨 Autenticador de Obras de Van Gogh")
st.markdown("Sube una imagen para predecir si una obra es **verdadera** o **falsa**.")

model_name = st.selectbox("Selecciona el modelo a usar:", list(MODEL_CONFIGS.keys()))
uploaded_file = st.file_uploader("📤 Sube tu imagen aquí", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Imagen subida", use_column_width=True)

    model = load_model(model_name)  # IMPORTANT: pass model_name
    transform = get_transform()

    st.write("⏳ Clasificando...")
    label, confidence, probs = predict_image(image, model, transform, CLASS_NAMES)

    st.success(f"🎯 Predicción: **{label}** con una confianza de **{confidence * 100:.2f}%**")

    st.subheader("Distribución de Probabilidades:")
    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, [p.item() * 100 for p in probs], color=["red", "green"])
    ax.set_ylabel("Probabilidad (%)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)

    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"**{class_name}:** {probs[i].item() * 100:.2f}%")

    try:
        log_prediction_to_mlflow(image, label, confidence, probs)
        st.info("✅ Resultado registrado en MLflow.")
    except Exception as e:
        st.warning(f"⚠️ No se pudo registrar en MLflow: {e}")

st.markdown("---")
st.info("Selecciona el modelo, sube una imagen y obtén la predicción. Los resultados se registran en MLflow si está disponible.")
st.caption("Desarrollado por Alex Reinoso - TFM")
