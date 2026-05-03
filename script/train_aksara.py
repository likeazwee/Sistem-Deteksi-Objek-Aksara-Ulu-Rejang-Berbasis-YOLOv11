from ultralytics import YOLO

def main():
    print("Melanjutkan Training Aksara Rejang")
    
    model = YOLO("models/best.pt")

    results = model.train(
        data="data.yaml",
        epochs=300,        
        patience=50,       
        imgsz=640,
        batch=8,           
        workers=2,         
        project="Aksara_Rejang",
        name="training_terbaru"
    )

if __name__ == '__main__':
    main()