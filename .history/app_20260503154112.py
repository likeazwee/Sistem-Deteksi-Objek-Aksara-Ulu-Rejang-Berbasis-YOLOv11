import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Konfigurasi Halaman Web
st.set_page_config(page_title="Deteksi Aksara Ulu Rejang", page_icon="🔍", layout="centered")

st.title("🔍 Deteksi Aksara Ulu Rejang")
st.write("Aplikasi pendeteksi Aksara Ulu Rejang berbasis AI menggunakan arsitektur YOLOv11.")
st.markdown("---")

# 2. Fungsi untuk memuat model (menggunakan cache agar tidak diload berulang kali)
@st.cache_resource
def load_model():
    # Pastikan path ini mengarah ke model terbaik dari Sesi 2 Anda
    return YOLO(r"runs\detect\Aksara_Rejang\training_v2_lanjutan\weights\best.pt")

model = load_model()

# 3. Area untuk mengunggah gambar
uploaded_file = st.file_uploader("Unggah gambar Aksara Rejang di sini...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar asli yang diunggah
    image = Image.open(uploaded_file)
    
    # Buat dua kolom untuk membandingkan sebelum & sesudah
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gambar Asli")
        st.image(image, use_container_width=True)

    # Tombol untuk mengeksekusi AI
    if st.button("Deteksi Aksara", use_container_width=True):
        with st.spinner('AI sedang menganalisis gambar...'):
            # Ubah gambar menjadi format yang bisa dibaca YOLO (numpy array RGB)
            img_array = np.array(image.convert("RGB"))
            
            # Lakukan prediksi (conf=0.5 berarti hanya tampilkan tebakan yang akurasinya di atas 50%)
            results = model.predict(source=img_array, conf=0.5)
            
            # Ambil gambar hasil prediksi beserta kotak (bounding box)
            res_plotted = results[0].plot()
            
            # YOLO mengembalikan gambar dalam format BGR, kita ubah kembali ke RGB untuk Streamlit
            res_img = Image.fromarray(res_plotted[..., ::-1])
            
            with col2:
                st.subheader("Hasil Deteksi")
                st.image(res_img, use_container_width=True)
                
        st.success("Deteksi selesai!")