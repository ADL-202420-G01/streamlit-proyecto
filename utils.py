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

# Definición del mapa de colores RGB para cada clase
colors_rgb = [
    (226, 169, 41),  # Water (#E2A929)
    (132, 41, 246),  # Land (unpaved area) (#8429F6)
    (110, 193, 228), # Road (#6EC1E4)
    (60, 16, 152),   # Building (#3C1098)
    (254, 221, 58),  # Vegetation (#FEDD3A)
    (155, 155, 155)  # Unlabeled (#9B9B9B)
]
labels = ["Agua", "Tierra", "Camino", "Edificios", "Vegetacion", "Sin etiqueta"]

def class_to_rgb(mask):
    """Convert class indices to RGB colors using a predefined color map."""
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(colors_rgb):
        rgb_mask[mask == i] = color
    return rgb_mask

def create_color_legend(colors, labels):
    """Crea una figura con una leyenda de colores."""
    fig, ax = plt.subplots(figsize=(2, 2))  # Ajusta el tamaño según necesites
    for i, (color, label) in enumerate(zip(colors, labels)):
        ax.barh(i, 1, color=np.array(color)/255.0, label=label)
    ax.set_xlim(0, 1)
    ax.legend(ncol=1, bbox_to_anchor=(1, 1), loc='upper left')
    plt.gca().invert_yaxis()
    plt.axis('off')  # Oculta los ejes
    return fig

def show_color_legend():
    # Crear la leyenda
    legend_fig = create_color_legend(colors_rgb, labels)

    # Mostrar la leyenda en Streamlit
    #st.pyplot(legend_fig)
    # Agregar descripciones de color en el sidebar o debajo de las imágenes
    st.sidebar.header("Leyenda de Colores")
    st.sidebar.text("🟨 Agua (rios, lagos): Amarillo #E2A929")
    st.sidebar.text("🟪 Suelo (area sin pavimento): Morado #8429F6")
    st.sidebar.text("🟦 Carretera: Azul claro #6EC1E4")
    st.sidebar.text("🟣 Construccion: Púrpura #3C1098")
    st.sidebar.text("🟩 Vegetacion: Verde #FEDD3A")
    st.sidebar.text("⬜ Unlabeled: Gris #9B9B9B")

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

