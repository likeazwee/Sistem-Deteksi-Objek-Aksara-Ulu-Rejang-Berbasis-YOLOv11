from ultralytics import YOLO

def main():
    # 1. Memuat model bawaan YOLOv11 versi Nano (paling ringan dan cepat)
    model = YOLO("yolo11n.pt")

    # 2. Memulai proses training
    results = model.train(
        data="data.yaml",  # Pastikan file data.yaml berada di folder yang sama dengan script ini
        epochs=100,        # Direkomendasikan 100 karena jumlah kelasnya banyak (253)
        imgsz=640,         # Standar resolusi YOLO
        batch=8,
        
        project="Aksara_Rejang", # Nama folder utama untuk menyimpan hasil
        name="training_v1"       # Nama sub-folder percobaan pertama
    )

if __name__ == '__main__':
    main()