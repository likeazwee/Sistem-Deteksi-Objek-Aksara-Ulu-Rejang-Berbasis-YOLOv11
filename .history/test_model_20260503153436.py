from ultralytics import YOLO

# 1. Memuat model yang baru saja selesai dilatih
# Pastikan path ini sesuai dengan lokasi file best.pt di laptop Anda
model = YOLO(r"runs\detect\Aksara_Rejang\training_v1-4\weights\best.pt")

# 2. Tentukan gambar yang mau dites
# Ganti nama file ini dengan salah satu gambar yang ada di folder 'test\images'
gambar_tes = r"test/images/3_jpg.rf.09612c2902eaf8ac9193bff9abd81b3c.jpg"
# 3. Jalankan prediksi (otomatis menampilkan dan menyimpan hasil gambar)
results = model.predict(source=gambar_tes, show=True, save=True)
