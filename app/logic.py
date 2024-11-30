import streamlit as st
import tensorflow as tf
import numpy as np
import sys
from PIL import Image

#funcion cargar
def load_model(path_model):
    model=tf.keras.models.load_model(path_model)
    return model

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