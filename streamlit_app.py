import streamlit as st
import tensorflow as tf
import numpy as np
import sys
from PIL import Image

#cargar el modelo guardado
model=tf.keras.models.load_model("Data/cnn_model.keras")

st.set_page_config(
    layout="centered", page_title="ADL 202420 Grupo 1", page_icon="🛩️"
)

#Funcion para preprocesar la imagen
def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Crear la interfaz en Streamlit
st.title("Clasificación con CNN")
st.write("Sube una imagen para que el modelo haga una predicción")

# Subir la imagen
uploaded_image = st.file_uploader("Eliga una imagen...", type=["jpg", "png"])

if uploaded_image is not None:
    # mostrar la imagen subida
    imagen = Image.open(uploaded_image)
    st.image(imagen, caption="Imagen Cargada", use_container_width=True)

    # preprocesar la imagen
    image_preprocessed = preprocess_image(imagen)

    # hacer la prediccion
    prediction = model.predict(image_preprocessed)
    predicted_class = np.argmax(prediction)
    predicted_label = label_map[predicted_class]

    #mostrar la prediccion
    st.write(f"Prediccion: {predicted_label}")

# Mostrar la versión de Python
st.write(f"Versión de Python: {sys.version}")
st.write(f"Versión de Streamlit: {st.__version__}. Versión de TensorFlow: {tf.__version__}. Versión de NumPy: {np.__version__}.")