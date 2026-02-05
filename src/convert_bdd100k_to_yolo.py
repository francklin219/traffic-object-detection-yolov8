import json
from pathlib import Path
from PIL import Image


BDD_ROOT = Path(r"C:\Users\franc\.cache\kagglehub\datasets\marquis03\bdd100k\versions\1")

TRAIN_JSON = BDD_ROOT / "train/annotations/bdd100k_labels_images_train.json"
VAL_JSON   = BDD_ROOT / "val/annotations/bdd100k_labels_images_val.json"

TRAIN_IMG = BDD_ROOT / "train/images"
VAL_IMG   = BDD_ROOT / "val/images"


OUT_ROOT = Path("data/bdd100k")
OUT_LABELS_TRAIN = OUT_ROOT / "labels/train"
OUT_LABELS_VAL   = OUT_ROOT / "labels/val"

OUT_LABELS_TRAIN.mkdir(parents=True, exist_ok=True)
OUT_LABELS_VAL.mkdir(parents=True, exist_ok=True)


CLASSES = [
    "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

def convert(json_file, img_dir, out_dir):
    data = json.loads(json_file.read_text(encoding="utf-8"))
    total_boxes = 0

    for item in data:
        img_name = item["name"]
        img_path = img_dir / img_name
        if not img_path.exists():
            continue

        w, h = Image.open(img_path).size
        yolo_lines = []

        for lab in item.get("labels", []):
            if "box2d" not in lab:
                continue

            cls = lab["category"]
            if cls not in CLASS_TO_ID:
                continue

            box = lab["box2d"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            yolo_lines.append(
                f"{CLASS_TO_ID[cls]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            )
            total_boxes += 1

        out_file = out_dir / f"{Path(img_name).stem}.txt"
        out_file.write_text("\n".join(yolo_lines), encoding="utf-8")

    print(f"✅ {json_file.name} → {total_boxes} boxes")

def main():
    convert(TRAIN_JSON, TRAIN_IMG, OUT_LABELS_TRAIN)
    convert(VAL_JSON, VAL_IMG, OUT_LABELS_VAL)

if __name__ == "__main__":
    main()
