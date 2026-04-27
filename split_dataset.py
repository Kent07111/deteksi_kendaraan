import random
import shutil
from pathlib import Path

base = Path(r"C:\Users\ASUS TUF\Documents\perkuliahan\semester 6\pengolahan citra digital\deteksi_kendaraan")

train_img = base / "train/images"
train_lbl = base / "train/labels"

images = list(train_img.glob("*"))
random.shuffle(images)

n = len(images)

valid_split = images[int(0.8*n):int(0.9*n)]
test_split = images[int(0.9*n):]

for split in ["valid", "test"]:
    (base / split / "images").mkdir(parents=True, exist_ok=True)
    (base / split / "labels").mkdir(parents=True, exist_ok=True)

for split_name, img_list in {
    "valid": valid_split,
    "test": test_split
}.items():
    for img in img_list:
        label = train_lbl / f"{img.stem}.txt"

        shutil.copy(img, base / split_name / "images" / img.name)

        if label.exists():
            shutil.copy(label, base / split_name / "labels" / label.name)

print("Split selesai.")
print("Total gambar:", n)
print("Valid:", len(valid_split))
print("Test:", len(test_split))