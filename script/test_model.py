from ultralytics import YOLO

model = YOLO("models/best.pt")

gambar_tes = r"test/images/3_jpg.rf.09612c2902eaf8ac9193bff9abd81b3c.jpg"
results = model.predict(source=gambar_tes, show=True, save=True)
