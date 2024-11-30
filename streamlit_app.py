import streamlit as st
import tensorflow as tf
import numpy as np
import sys
from PIL import Image

from app import logic

st.set_page_config(
    layout="centered", page_title="ADL 202420 Grupo 1", page_icon="🛩️"
)

# Crear la interfaz en Streamlit
st.title("Localización de construcciones con imágenes satelitales")
st.write("Sube una imagen satelital para identificar construcciones")

# Subir la imagen
uploaded_image = st.file_uploader("Eliga una imagen...", type=["jpg", "png"])

MODEL_PATH = "cnn_model.keras"
model = logic.load_model(MODEL_PATH)

if model is not None:
    st.write("Resumen del modelo:")
    st.text(model.summary())
    
if uploaded_image is not None:
    # Mostrar la imagen cargada
    original_image = Image.open(uploaded_image)
    st.image(original_image, caption="Imagen Original", use_column_width=True)

    # Preprocesar la imagen
    input_image = logic.preprocess_image(original_image, target_size=(128, 128))  # Ajusta según el modelo

    # Pasar la imagen por el modelo
    prediction = model.predict(input_image)

    # Postprocesar la salida
    result_image = logic.postprocess_output(prediction)

    # Mostrar la imagen procesada
    st.image(result_image, caption="Imagen Procesada", use_column_width=True)

# Mostrar información adicional
st.write("Procesamiento completo.")

# Mostrar la versión de Python
st.write(f"Versión de Python: {sys.version}")
st.write(f"Versión de SKetreamlit: {st.__version__}. Versión de NumPy: {np.__version__}.")
st.write(f"Versión de TensorFlow: {tf.__version__}. Versión de Keras: {tf.keras.__version__}.")
st.write(f"Autores: Edison Suarez, Nicolas Niño, Diego Noriega, Freddy Freddy Orjuela")