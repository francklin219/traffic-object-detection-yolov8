from ultralytics import YOLO

def main():
    
    model = YOLO("yolov8n.pt")

    model.train(
        data="data/bdd100k.yaml",
        imgsz=640,
        epochs=20,
        batch=8,
        device="cpu",     
        project="runs",
        name="bdd100k_yolov8n"
    )

if __name__ == "__main__":
    main()
