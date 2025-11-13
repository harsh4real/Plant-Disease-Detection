import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# ------------------------------
# Load class names
# ------------------------------
with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

# ------------------------------
# Load Keras model (compatible with TensorFlow 2.20)
# ------------------------------
MODEL_PATH = "plant_disease_saved_model"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ------------------------------
# Disease Descriptions
# ------------------------------
DISEASE_INFO = {
    "Healthy": {
        "description": "The leaf shows no disease symptoms and appears healthy.",
        "treatment": "No treatment needed. Maintain good nutrition, sunlight, and watering."
    },
    "Powdery": {
        "description": "Powdery Mildew: White powder-like fungal growth on leaves.",
        "treatment": "Use neem oil spray, increase airflow, avoid overhead watering, remove infected leaves."
    },
    "Rust": {
        "description": "Leaf Rust: Yellow/orange rust-like fungal spots.",
        "treatment": "Use sulfur fungicide, prune infected leaves, avoid wetting foliage frequently."
    }
}

# ------------------------------
# Preprocessing Function
# ------------------------------
def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return tf.convert_to_tensor(image, dtype=tf.float32)

# ------------------------------
# Prediction Function
# ------------------------------
def predict(image):
    processed = preprocess(image)
    probs = model(processed, training=False).numpy()[0]

    class_index = np.argmax(probs)
    class_name = CLASS_NAMES[class_index]
    confidence = float(np.max(probs)) * 100

    return class_name, confidence, probs


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="centered")

st.title("🌿 Plant Disease Detection App")
st.write("Upload a plant leaf image to detect whether it is **Healthy**, affected by **Powdery Mildew**, or **Rust**.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)

    st.write("---")

    st.subheader("🔍 Prediction Result")
    class_name, confidence, probs = predict(image)

    st.success(f"### 🌱 Disease: **{class_name}**")
    st.info(f"### 🔢 Confidence: **{confidence:.2f}%**")

    st.write("---")

    st.subheader("📘 Disease Information")
    st.write(f"**Description:** {DISEASE_INFO[class_name]['description']}")
    st.write(f"**Treatment:** {DISEASE_INFO[class_name]['treatment']}")

    st.write("---")

    st.subheader("📊 Probability Breakdown")
    for i, name in enumerate(CLASS_NAMES):
        st.write(f"{name}: {probs[i] * 100:.2f}%")
else:
    st.info("📥 Upload an image to get started.")



