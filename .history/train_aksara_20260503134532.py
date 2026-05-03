from ultralytics import YOLO

# 1. Gunakan 'otak' terbaik dari Sesi 1 sebagai pondasi
model = YOLO(r"runs\detect\Aksara_Rejang\training_v1-4\weights\best.pt")

# 2. Mulai Sesi 2 dengan target epoch besar
results = model.train(
    data="data.yaml",
    epochs=300,        # Pasang angka besar sekaligus
    patience=50,       # PENTING: Jika dalam 50 putaran akurasi tidak naik sama sekali, YOLO akan otomatis stop!
    imgsz=640,
    batch=8,
    workers=2,
    project="Aksara_Rejang",
    name="training_v2_lanjutan" # Simpan di folder baru agar grafik Sesi 1 tidak tertimpa
)

if __name__ == '__main__':
    main()