import streamlit as st

### Support functions
def generate_progress_bar(value):
    return f'<div style="width: 100%; border: 1px solid #eee; border-radius: 10px;"><div style="width: {value * 100}%; height: 24px; background: linear-gradient(90deg, rgba(62,149,205,1) 0%, rgba(90,200,250,1) 100%); border-radius: 10px;"></div></div>'

# Main function
def main():
    st.set_page_config(
    layout="centered", page_title="Analisis con Deep Learning", page_icon="🛩️"
)
    # UI
    st.title('Detección de Construcciones con Imágenes Satelitales')
    # Texto introductorio
    st.markdown("""
    Bienvenido a la herramienta de localizacion de construcciones.
                """)
    # Subir archivo 
    uploaded_file = st.file_uploader("Subir imagen", type=["jpg"])


# Main function call to run the program
if __name__ == '__main__':
    main()