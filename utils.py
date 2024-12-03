import matplotlib.pyplot as plt
import numpy as np
import sys
import streamlit as st
import tensorflow as tf

from PIL import Image

def load_image(image_file):
    """Carga una imagen desde un archivo y la convierte en un array de numpy."""
    image = Image.open(image_file)
    image = image.resize((256, 256))  # Redimensionar si es necesario
    return np.array(image)

def preprocess_image(image):
    """Aplica preprocesamientos necesarios antes de la predicción."""
    return image / 255.0  # Normalización simple

"""
# Definición del mapa de colores RGB para cada clase
colors_rgb = [
    (60, 16, 152),   # Building (#3C1098) Púrpura
    (132, 41, 246),  # Land (unpaved area) (#8429F6) Morado
    (110, 193, 228), # Road (#6EC1E4)
    (254, 221, 58),  # Vegetation (#FEDD3A) Amarillo
    (226, 169, 41),  # Water (#E2A929) Naranja
    (155, 155, 155)  # Unlabeled (#9B9B9B) Gris
]
"""
def class_to_rgb(mask, config):
    """Convert class indices to RGB colors using a predefined color map from config."""
    classes = config.get("classes", [])
    # Crear un diccionario para mapear los índices de clase a colores RGB
    class_to_color = {i: (int(cls['color'][1:3], 16), int(cls['color'][3:5], 16), int(cls['color'][5:7], 16)) for i, cls in enumerate(classes)}
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i, color in class_to_color.items():
        rgb_mask[mask == i] = color
    return rgb_mask

def display_color_legend(config):
    st.sidebar.header("Leyenda de Colores")
    classes = config.get("classes", [])
    if not classes:
        st.sidebar.write("No hay información de clases disponible.")
        return

    for item in classes:
        color = item['color']
        description = item['label']
        color_box = f"<span style='display:inline-block; width:12px; height:12px; background-color:{color};'></span>"
        st.sidebar.markdown(f"{color_box} {description}", unsafe_allow_html=True)

def get_versions():
    """Devuelve una cadena con las versiones de las bibliotecas importantes."""
    python_version = sys.version
    streamlit_version = st.__version__
    numpy_version = np.__version__
    tensorflow_version = tf.__version__
    keras_version = tf.keras.__version__

    return (f"Versión de Python: {python_version}\n"
            f"Versión de Streamlit: {streamlit_version}. Versión de NumPy: {numpy_version}.\n"
            f"Versión de TensorFlow: {tensorflow_version}. Versión de Keras: {keras_version}.")

def get_authors():
    """Devuelve una cadena con los nombres de los autores."""
    return "Autores: Edison Suarez, Nicolas Niño, Diego Noriega, Freddy Orjuela"

