import kagglehub
import shutil
from pathlib import Path


print("⬇️ Downloading BDD100K from Kaggle...")
src_path = Path(kagglehub.dataset_download("marquis03/bdd100k"))

print("📂 Dataset downloaded at:", src_path)


dst_images = Path("data/bdd100k/images")
dst_images.mkdir(parents=True, exist_ok=True)


images_100k = src_path / "bdd100k" / "images" / "100k"

if not images_100k.exists():
    raise FileNotFoundError("❌ images/100k not found in downloaded dataset")

print("📁 Copying images...")

shutil.copytree(images_100k / "train", dst_images / "train", dirs_exist_ok=True)
shutil.copytree(images_100k / "val",   dst_images / "val",   dirs_exist_ok=True)

print("✅ Images copied to data/bdd100k/images")