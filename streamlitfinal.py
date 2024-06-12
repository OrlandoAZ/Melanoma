import streamlit as st
import os
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

# Cargar el modelo de Keras
model = load_model("keras_model.h5", compile=False)

# Cargar las etiquetas
class_names = open("labels.txt", "r").readlines()

# Crear el directorio temporal si no existe
temp_dir = "temp"
os.makedirs(temp_dir, exist_ok=True)

# Función de preprocesamiento y clasificación de la imagen
def clasificar_imagen(imagen_path):
    image = Image.open(imagen_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    pred = model.predict(data)[0]
    return pred

# Configuración para desactivar la advertencia
st.set_option('deprecation.showPyplotGlobalUse', False)

# Encabezado
st.title('MODELO MELANOMAS EN HUMANOS')
st.markdown("Aplicación elaborada por Orlando Advíncula Zeballos.")
# Sección de carga de imágenes
uploaded_file = st.file_uploader("Selecciona la imagen de un melanoma", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Guardar temporalmente el archivo
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Mostrar la imagen
    lena_rgb = io.imread(temp_path) / 255.0

    # Crear una figura de Matplotlib y pasarla a st.pyplot()
    fig, ax = plt.subplots()
    ax.imshow(lena_rgb)
    ax.set_title("Imagen seleccionada")
    ax.axis('off')
    st.pyplot(fig)

    # Clasificar la imagen
    pred = clasificar_imagen(temp_path)

    # Mostrar resultado de la clasificación
    class_index = np.argmax(pred)
    class_name = class_names[class_index].strip()
    confidence_score = pred[class_index]

    # Estilos personalizados
    tipo_melanoma_style = f"font-size: 40px; color: {'red' if class_index == 1 else 'green'}"
    probabilidad_style = "font-size: 40px;"
    #0 benign
    #1 malignant
    
    # Mostrar resultado de la clasificación con estilos personalizados
    st.markdown(f'<p style="{tipo_melanoma_style}">Tipo de Melanoma: {class_name[2:]}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="{probabilidad_style}">Probabilidad: {confidence_score}</p>', unsafe_allow_html=True)

    # Eliminar el archivo temporal después de usarlo
    os.remove(temp_path)
