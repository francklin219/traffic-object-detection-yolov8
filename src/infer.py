from ultralytics import YOLO
from pathlib import Path

def main():
    weights = "runs/bdd100k_yolov8n/weights/best.pt"
    model = YOLO(weights)

    source = "assets/demo"  
    outdir = Path("assets/predictions")
    outdir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=source,
        conf=0.25,
        save=True,
        project=str(outdir),
        name="preds"
    )

    print("✅ Predictions saved in:", outdir / "preds")

if __name__ == "__main__":
    main()

