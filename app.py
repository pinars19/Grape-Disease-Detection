import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.title("Grape Disease Detection - CNN")
st.write("Üzüm yaprağı görüntüsü yükleyerek hastalık tespiti yapabilirsiniz.")

model = load_model("grape_cnn_v2.keras")

classes = ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]

file = st.file_uploader("Görüntü yükle", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB").resize((128, 128))
    st.image(img, caption="Yüklenen görüntü")

    x = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = model.predict(x, verbose=0)[0]

    st.success(f"Tahmin: {classes[np.argmax(pred)]}")
    st.write(f"Güven skoru: {np.max(pred):.2%}")
