import numpy as np
import streamlit as st
import tensorflow as tf
import os
import sys
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate, LayerNormalization, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MultiHeadAttention
from PIL import Image

# Función para preprocesar la imagen con soporte para target_size
def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)  # Cambiar tamaño según target_size
    image = np.array(image) / 255.0  # Normalización (opcional)
    image = np.expand_dims(image, axis=0)  # Añadir dimensión para lotes
    return image

# Función para postprocesar la salida del modelo
def postprocess_output(output):
    # Escalar la salida a valores entre 0-255 y convertirla en imagen
    output = np.squeeze(output)  # Eliminar dimensiones no necesarias
    output = (output * 255).astype(np.uint8)
    return Image.fromarray(output)

# Swin Transformer block
class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, embed_dim, **kwargs):
        super(SwinTransformerBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.norm1 = LayerNormalization(epsilon=1e-5)
        self.attn = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.norm2 = LayerNormalization(epsilon=1e-5)
        self.mlp = tf.keras.Sequential([
            Dense(self.embed_dim, activation=tf.nn.gelu),
            Dense(self.embed_dim)
        ])
    
    def call(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, x)
        x = shortcut + x
        x = self.norm2(x)
        x = self.mlp(x)
        return x

def multi_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Swin Transformer block
    swin_block = SwinTransformerBlock(num_heads=8, embed_dim=128)
    x = swin_block(p4)

    # Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Assuming metrics and loss are defined
metrics = ['accuracy']  # Replace 'jacard_coef' with the actual implementation if you have it
total_loss = 'categorical_crossentropy'  # Replace with the actual loss function if different

def get_model():
    return multi_unet_model(n_classes=6, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3)

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