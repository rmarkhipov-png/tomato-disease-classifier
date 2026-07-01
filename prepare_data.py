import os
import shutil
import random
from pathlib import Path

SOURCE = Path("C:/Users/curba/Desktop/AI/data/raw/plant-village/PlantVillage")
DEST = Path("C:/Users/curba/Desktop/AI/data/splits")

TOMATO_CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus"
]

random.seed(42)

for cls in TOMATO_CLASSES:
    src_dir = SOURCE / cls
    if not src_dir.exists():
        print(f"NOT FOUND: {cls}")
        continue
    images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.JPG")) + list(src_dir.glob("*.png")) + list(src_dir.glob("*.PNG"))
    random.shuffle(images)
    n = len(images)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }
    for split, files in splits.items():
        out_dir = DEST / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, out_dir / f.name)
    print(f"OK {cls}: {n} total, train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

print("DONE!")
