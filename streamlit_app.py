import streamlit as st
from model import load_model, predict
from utils import load_image, preprocess_image, class_to_rgb
from utils import show_color_legend, get_versions, get_authors

# Configuración de la página para usar todo el ancho disponible
st.set_page_config(
    layout="wide",  
    page_title="ADL 202420 Grupo 1",
    page_icon="🛰️"  # Ícono de satélite
)

# Crear la interfaz en Streamlit
st.title("Segmentacion de imagenes satelitales para catastro")
st.write("Modelo de apoyo para trabajos de catastro")

MODEL_PATH = "best_model.keras"
model = load_model()

# Subir la imagen
uploaded_file = st.file_uploader("Eliga una imagen jpg...", type=["jpg"])

if uploaded_file is not None:
    
    image = load_image(uploaded_file)
    image_processed = preprocess_image(image)
    prediction = predict(model, image_processed)
    mask_rgb = class_to_rgb(prediction) 

    col1, col2 = st.columns(2)  # Crear dos columnas

    with col1:
        st.write("Imagen Original")
        st.image(image, use_container_width=True)  # Mostrar imagen original en la primera columna

    with col2:
        st.write("Máscara Predicha")
        st.image(mask_rgb, use_container_width=True)  # Mostrar máscara predicha en la segunda columna

    # Crear la leyenda
    show_color_legend()
#else:
#    st.write("Por favor, carga una imagen para analizar.")

# Mostrar la versión de Python
versions_info = get_versions()
st.write(versions_info)

authors_info = get_authors()
st.write(authors_info)