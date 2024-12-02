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
#model = logic.get_model()
model = tf.keras.models.load_model(MODEL_PATH)
# Assuming metrics and loss are defined
metrics = ['accuracy']  # Replace 'jacard_coef' with the actual implementation if you have it
total_loss = 'categorical_crossentropy'  # Replace with the actual loss function if different

#if model is not None:
#    model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
#    model.load_weights(MODEL_PATH)
#    st.success("Modelo cargado exitosamente.")

# Subir la imagen
st.write("Cargue una imagen para subirlo al modelo")
uploaded_image = st.file_uploader("Eliga una imagen jpg...", type=["jpg"])

if uploaded_image is not None:
    # Mostrar la imagen cargada
    original_image = Image.open(uploaded_image)
    st.image(original_image, caption="Imagen Original", use_container_width=True)

    # Cambiar tamaño antes de preprocesar
    resized_image = original_image.resize((512, 512))

    # Preprocesar la imagen
    input_image = logic.preprocess_image(resized_image)  # Sin target_size

    # Pasar la imagen por el modelo
    prediction = model.predict(input_image)

    # Postprocesar la salida
    result_image = logic.postprocess_output(prediction)

    # Mostrar la imagen procesada
    st.image(result_image, caption="Imagen Procesada", use_column_width=True)

# Mostrar la versión de Python
st.write(f"Versión de Python: {sys.version}")
st.write(f"Versión de Streamlit: {st.__version__}. Versión de NumPy: {np.__version__}.")
st.write(f"Versión de TensorFlow: {tf.__version__}. Versión de Keras: {tf.keras.__version__}.")
st.write(f"Autores: Edison Suarez, Nicolas Niño, Diego Noriega, Freddy Orjuela")