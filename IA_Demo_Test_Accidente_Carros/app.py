import os
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import pandas as pd

# Título centrado con imagen de banner
st.markdown("<h1 style='text-align: center;'>CrashApp</h1>", unsafe_allow_html=True)
banner_image = Image.open("banner.png")
st.image(banner_image, use_column_width=True)

with st.expander("Cómo funciona CrashApp"):
    st.write("""
        CrashApp es una aplicación creada para Toyota para poder identificar los daños y costos que puedan tener los autos al ser importados.
        Esta aplicación utiliza tecnologías de visión por computador para poder identificar distintos daños, y luego en base a un modelo matemático estimar el costo de reparación.
        Cada sección tiene una ventana flotante en caso de dudas sobre cómo rellenar los campos.
    """)

st.write("---")  # Agrega una línea divisoria en la interfaz

# Sección para la carga de imágenes en una grilla de 2x2
st.subheader("Carga de imágenes")

# Carga de imágenes en una grilla de 2x2

uploaded_images = st.file_uploader("Cargar imágenes", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

# Campo de texto para ingresar el ID
input_id = st.text_input("Ingrese el UUID del auto")
# Lista de los 10 modelos de autos Toyota más vendidos (con '_' en lugar de espacios)
modelos_toyota = [
    "Corolla",
    "Camry",
    "RAV4",
    "Hilux",
    "Highlander",
    "Tacoma",
    "4Runner",
    "Tundra",
    "Sienna",
    "Prius"
]

# Campo de selección para modelos de autos
modelo_seleccionado = st.selectbox("Seleccione el modelo del auto", modelos_toyota)


if st.button('Correr modelo'):

    model_detections = []

    if not modelo_seleccionado:
        st.error("Por favor, seleccione un modelo de auto antes de correr el modelo.")
    elif not uploaded_images:
        st.error("Por favor, cargue al menos un archivo antes de correr el modelo.")
    elif not input_id:
        st.error("Por favor, ingrese el UUID del auto antes de correr el modelo.")
    elif uploaded_images:
        folder_name = f"{input_id}_{modelo_seleccionado}"
        # if exists, delete and create again
        if os.path.exists(folder_name):
            import shutil
            shutil.rmtree(folder_name)
        os.makedirs(folder_name)
        # create folder inside with name "input" to save all the images
        os.makedirs(os.path.join(folder_name, "input"))
        input_folder_name = os.path.join(folder_name, "input")
        for uploaded_image in uploaded_images:
            # save image in folder with name "{input_id}_{modelo_seleccionado}_{image_number}"
            image_number = uploaded_images.index(uploaded_image)
            image_path = os.path.join(input_folder_name, f"{input_id}_{modelo_seleccionado}_{image_number}.jpg")
            input_file_name = f"{input_id}_{modelo_seleccionado}_{image_number}"
            with open(image_path, 'wb') as f:
                f.write(uploaded_image.getbuffer())
        # send images to API
            files = {'file': (uploaded_image.name, uploaded_image.getvalue())}
            response = requests.post('http://localhost:8000/api/predict', files=files, data={'input_id': input_id,'input_file_name': input_file_name, 'modelo': modelo_seleccionado})
            if response.status_code == 200:
                data = response.json()
                image_path = data['image_url']
                detections = data['detections']
                for det in detections:
                    model_detections.append(det['label'])
                    print(f"Label: {det['label']}, Confidence: {det['confidence']}")
                    #st.write(f"Label: {det['label']}, Confidence: {det['confidence']}")
                st.image(image_path, caption='Imagen procesada', use_column_width=True)
            else:
                st.error("Error al procesar la imagen")


        damage_prices = {
            "dent": 350000,
            "scratch": 150000,
            "crack": 200000,
            "glass shatter": 500000,
            "lamp broken": 250000,
            "tire flat": 100000
        }
        # PRINT MESSAGE WITH A COUNT OF EACH DETECTION TYPE
        if model_detections:
            summary_data = {
                "Detección": [],
                "Cantidad": [],
                "Costo Total": []
            }
            for det in set(model_detections):
                cantidad = model_detections.count(det)
                costo_total = cantidad * damage_prices[det]
                summary_data["Detección"].append(det)
                summary_data["Cantidad"].append(cantidad)
                summary_data["Costo Total"].append(costo_total)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df["Costo Total"] = summary_df["Costo Total"].apply(lambda x: f"${x:,.0f}")
            st.write("Resumen de detecciones y costos:")
            st.table(summary_df)
     #       st.write("Resumen de detecciones:")
     #       for det in set(model_detections):
     #           st.write(f"{det}: {model_detections.count(det)}")
    else:
        st.warning("Por favor, carga alguna imagen antes de correr el modelo.")
 #   else:

#cols = st.columns(2)
#image_slots = [col.empty() for col in cols for _ in range(2)]
#image_uploaders = [col.file_uploader("Cargar imagen", type=['jpg', 'jpeg', 'png'], key=f"image_{i}") for i, col in enumerate(image_slots)]

# Mostrar las imágenes cargadas directamente bajo cada uploader
#for image_slot, image_uploader in zip(image_slots, image_uploaders):
#    if image_uploader is not None:
#        image_slot.image(image_uploader, use_column_width=True)

#st.write("---")
#API_URL = os.environ.get('API_URL', 'http://127.0.0.1:8000')  # Default a localhost para desarrollo
#if st.button('Correr modelo'):
#    response = requests.get(f'{API_URL}/api/hello')
#    if response.status_code == 200:
#        data = response.json()
#        st.text(data['message'])
#    else:
#        st.error(f'Error al obtener respuesta del servidor {API_URL} ({response.status_code}) {response.text}')
