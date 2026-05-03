import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Deteksi Aksara Ulu Rejang", page_icon="🔍", layout="centered")

st.title("Deteksi Aksara Ulu Rejang")
st.write("Aplikasi pendeteksi Aksara Ulu Rejang berbasis AI menggunakan arsitektur YOLOv11.")
st.markdown("---")

@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

uploaded_file = st.file_uploader("Unggah gambar Aksara Rejang di sini...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gambar Asli")
        st.image(image, use_container_width=True)

    if st.button("Deteksi Aksara", use_container_width=True):
        with st.spinner('AI sedang menganalisis gambar...'):
            img_array = np.array(image.convert("RGB"))

            results = model.predict(source=img_array, conf=0.5)

            res_plotted = results[0].plot()

            res_img = Image.fromarray(res_plotted[..., ::-1])
            
            with col2:
                st.subheader("Hasil Deteksi")
                st.image(res_img, use_container_width=True)
                
        st.success("Deteksi selesai!")