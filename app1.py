import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.title("Grape Disease Detection - MobileNetV2")
st.write("Transfer Learning ile üzüm yaprağı hastalık tespiti")

model = load_model("grape_mobilenet.keras")

classes = ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]

file = st.file_uploader("Görüntü yükle", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB").resize((128, 128))
    st.image(img, caption="Yüklenen görüntü")

    x = np.expand_dims(np.array(img, dtype="float32"), axis=0)
    x = preprocess_input(x)

    pred = model.predict(x, verbose=0)[0]

    st.success(f"Tahmin: {classes[np.argmax(pred)]}")
    st.write(f"Güven skoru: {np.max(pred):.2%}")
