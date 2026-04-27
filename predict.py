from ultralytics import YOLO

model = YOLO(r"runs/detect/detection/weights/best.pt")

model.predict(
    source="gambar_uji6.jpg",
    conf=0.25,
    save=True,
    device=0,
    save_dir="runs/detect/predict"  
)