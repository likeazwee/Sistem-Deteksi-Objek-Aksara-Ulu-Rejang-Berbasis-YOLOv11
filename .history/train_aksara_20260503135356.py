from ultralytics import YOLO

def main():
    print("🚀 Melanjutkan Training Aksara Rejang - Sesi 2")
    
    # 1. Gunakan 'best.pt' dari hasil training sebelumnya (v1) sebagai dasar
    # Pastikan path ini benar sesuai lokasi folder runs Anda
    model = YOLO(r"runs\detect\Aksara_Rejang\training_v1\weights\best.pt")

    # 2. Mulai training lanjutan dengan target epoch lebih banyak
    results = model.train(
        data="data.yaml",
        epochs=300,        # Kita pasang 300 epoch sekalian
        patience=50,       # Otomatis berhenti jika dalam 50 epoch tidak ada kemajuan
        imgsz=640,
        batch=8,           # Tetap 8 agar aman di VRAM 6GB RTX 4050
        workers=2,         # Tetap 2 agar aman dari Error 1455 di Windows
        project="Aksara_Rejang",
        name="training_v2_lanjutan"
    )

# WAJIB: Pintu masuk utama agar multiprocessing (workers) berjalan lancar di Windows
if __name__ == '__main__':
    main()