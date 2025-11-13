import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model and classes
@st.cache_resource
def load_model():
    model = tf.saved_model.load("plant_disease_saved_model")
    infer = model.signatures["serving_default"]  # Keras 3.x SavedModel interface

    with open("class_names.json") as f:
        class_names = json.load(f)

    return infer, class_names

infer, class_names = load_model()

# Streamlit UI
st.title("🌿 Plant Leaf Disease Detection")
st.write("Upload a plant leaf image and the model will predict the disease.")

uploaded = st.file_uploader("Upload JPG/PNG Image", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_resized = img.resize((224, 224))
    x = np.array(img_resized, dtype=np.float32)[None, ...]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    outputs = infer(tf.constant(x))
    preds = list(outputs.values())[0].numpy()

    idx = np.argmax(preds[0])
    confidence = preds[0][idx] * 100

    st.subheader(f"Prediction: **{class_names[idx]}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
