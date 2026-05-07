from ultralytics import YOLO

model = YOLO("models/best.pt")

gambar_tes = r"gambar_dari_dataset"
results = model.predict(source=gambar_tes, show=True, save=True)
