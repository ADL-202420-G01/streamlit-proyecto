import streamlit as st
from model import load_model, predict
from utils import load_image, preprocess_image, create_mask_overlay


st.set_page_config(
    layout="centered", page_title="ADL 202420 Grupo 1", page_icon="🛩️"
)

# Crear la interfaz en Streamlit
st.title("Segmentacion de imagenes satelitales para catastro")
st.write("Modelo de apoyo para trabajos de catastro")

MODEL_PATH = "best_model.keras"
model = load_model()
# model = tf.keras.models.load_model(MODEL_PATH)
# Assuming metrics and loss are defined
metrics = ['accuracy']  # Replace 'jacard_coef' with the actual implementation if you have it
total_loss = 'categorical_crossentropy'  # Replace with the actual loss function if different

#if model is not None:
#    model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
#    model.load_weights(MODEL_PATH)
#    st.success("Modelo cargado exitosamente.")

# Subir la imagen
uploaded_image = st.file_uploader("Eliga una imagen jpg...", type=["jpg"])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    image_processed = preprocess_image(image)
    prediction = predict(model, image_processed)
    overlay_image = create_mask_overlay(image, prediction)
    
    st.image(image, caption='Imagen Original', use_column_width=True)
    st.image(overlay_image, caption='Predicción de Segmentación', use_column_width=True)

# Mostrar la versión de Python
st.write(f"Versión de Python: {sys.version}")
st.write(f"Versión de Streamlit: {st.__version__}. Versión de NumPy: {np.__version__}.")
st.write(f"Versión de TensorFlow: {tf.__version__}. Versión de Keras: {tf.keras.__version__}.")
st.write(f"Autores: Edison Suarez, Nicolas Niño, Diego Noriega, Freddy Orjuela")