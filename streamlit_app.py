import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

#from src.back.ModelController import ModelController

st.set_page_config(
    layout="centered", page_title="ADL 2024 Grupo 1", page_icon="🛩️"
)

### Support functions
def generate_progress_bar(value):
    return f'<div style="width: 100%; border: 1px solid #eee; border-radius: 10px;"><div style="width: {value * 100}%; height: 24px; background: linear-gradient(90deg, rgba(62,149,205,1) 0%, rgba(90,200,250,1) 100%); border-radius: 10px;"></div></div>'

#ctrl = ModelController()

# UI
st.title('Detección de Contrucciones con Imágenes Satelitales')
# Texto introductorio
st.markdown("""
Bienvenido a la herramienta. Esta aplicación permite localizar contrucciones usando Técnicas supervisadas de Deep Learning. 

## ¿Cómo funciona?
1. Suba una imagen satelital de las tomadas en campo.
2. La aplicación muestra la imagen que se quiere analizar.
3. De clic en 'localizar' para que muestre los resultados.
""")
# Subir la imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Leer la imagen usando Pillow
    image = Image.open(uploaded_file)

    # Mostrar la imagen en la aplicación
    st.image(image, caption="Imagen subida", use_container_width=True)
else:
    st.write("Por favor, sube una imagen para mostrarla.")
