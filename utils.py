import numpy as np
from PIL import Image

def load_image(image_file):
    """Carga una imagen desde un archivo y la convierte en un array de numpy."""
    image = Image.open(image_file)
    image = image.resize((256, 256))  # Redimensionar si es necesario
    return np.array(image)

def preprocess_image(image):
    """Aplica preprocesamientos necesarios antes de la predicción."""
    return image / 255.0  # Normalización simple

def create_mask_overlay(image, mask):
    """Crea una superposición de máscara sobre la imagen original."""
    # Suponiendo que mask es una imagen binaria [0, 1]
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = [255, 0, 0]  # Rojo para la máscara
    overlay = ((image * 0.7) + (colored_mask * 0.3)).astype("uint8")
    return overlay
