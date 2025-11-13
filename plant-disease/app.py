import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ------------------------------
# Load class names
# ------------------------------
with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

# ------------------------------
# Load SavedModel (TensorFlow 2.20 format)
# ------------------------------
MODEL_PATH = "plant_disease_saved_model"
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]

# ------------------------------
# Disease Descriptions
# ------------------------------
DISEASE_INFO = {
    "Healthy": {
        "description": "The leaf appears healthy with no visible disease symptoms.",
        "treatment": "No treatment is required. Maintain good watering, sunlight, and nutrition."
    },
    "Powdery": {
        "description": "Powdery Mildew: A fungal disease that appears as white powdery spots on leaves.",
        "treatment": "Use neem oil spray, remove affected leaves, improve airflow, avoid overhead watering."
    },
    "Rust": {
        "description": "Leaf Rust: A fungal disease that causes yellow/orange rust-like spots.",
        "treatment": "Use sulfur-based fungicides, remove infected leaves, avoid wetting leaves frequently."
    }
}

# ------------------------------
# Preprocessing Function
# ------------------------------
def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    return image

# ------------------------------
# Prediction Function
# ------------------------------
def predict(image):
    processed = preprocess(image)
    output = infer(processed)

    probs = list(output.values())[0].numpy()[0]
    class_index = np.argmax(probs)
    class_name = CLASS_NAMES[class_index]
    confidence = float(np.max(probs)) * 100

    return class_name, confidence, probs


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="centered")

st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image and the model will detect whether it is **Healthy**, has **Powdery Mildew**, or **Rust**.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, caption="Input Leaf", use_column_width=True)

    st.write("---")

    st.subheader("🔍 Prediction Result")
    class_name, confidence, probs = predict(image)

    st.success(f"### 🌱 Predicted Disease: **{class_name}**")
    st.info(f"### 🔢 Confidence: **{confidence:.2f}%**")

    st.write("---")

    # Additional information
    st.subheader("📘 Disease Information")
    st.write(f"**Description:** {DISEASE_INFO[class_name]['description']}")
    st.write(f"**Treatment Advice:** {DISEASE_INFO[class_name]['treatment']}")

    st.write("---")

    # Show probability distribution
    st.subheader("📊 Prediction Probabilities")
    for i, cname in enumerate(CLASS_NAMES):
        st.write(f"{cname}: {probs[i] * 100:.2f}%")

else:
    st.info("📤 Please upload an image to begin.")


