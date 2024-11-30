import numpy as np
import streamlit as st
import tensorflow as tf
import os
import sys
from PIL import Image

#funcion cargar
def load_model(model_path):
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            st.success("Modelo cargado exitosamente.")
            return model
        except ValueError as ve:
            st.error(f"Error al cargar el modelo. Verifica la compatibilidad: {ve}")
        except Exception as e:
            st.error(f"Ocurrió un error inesperado: {e}")
    else:
        st.error(f"El archivo {model_path} no existe. Asegúrate de que la ruta sea correcta.")
    
    return None

#Funcion para preprocesar la imagen
def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Función para postprocesar la salida del modelo
def postprocess_output(output):
    # Escalar la salida a valores entre 0-255 y convertirla en imagen
    output = np.squeeze(output)  # Eliminar dimensiones no necesarias
    output = (output * 255).astype(np.uint8)
    return Image.fromarray(output)