import io
import math
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm.std import tqdm

REPO = "benjamin-paine/imagenet-1k-128x128"
LABELS = [0, 1, 2, 3, 4, 5, 6, 147, 148]
TOTAL = 10_000
OUT_128 = Path("images/whale_fish_128")
OUT_64 = Path("images/whale_fish_64")
OUT_128.mkdir(parents=True, exist_ok=True)
OUT_64.mkdir(parents=True, exist_ok=True)

max_per_class = math.ceil(TOTAL / len(LABELS))
class_counts = {l: 0 for l in LABELS}
saved = 0
pbar = tqdm(total=TOTAL, desc="Saving whale/fish images")

for shard_idx in range(13):
    if saved >= TOTAL:
        break
    shard_file = f"data/train-{shard_idx:05d}-of-00013.parquet"
    print(f"\n[Shard {shard_idx + 1}/13] {shard_file}")
    path = hf_hub_download(
        repo_id=REPO, repo_type="dataset", filename=shard_file,
        token=os.getenv("HF_TOKEN"),
    )
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=1000, columns=["label", "image"]):
        labels = batch.column("label").to_pylist()
        images = batch.column("image")
        for i, label in enumerate(labels):
            if saved >= TOTAL:
                break
            if label in class_counts and class_counts[label] < max_per_class:
                img = Image.open(io.BytesIO(images[i]["bytes"].as_py())).convert("RGB")
                if img.size != (128, 128):
                    img = img.resize((128, 128), Image.LANCZOS)
                name = f"{saved:05d}.png"
                img.save(OUT_128 / name)
                img.resize((64, 64), Image.LANCZOS).save(OUT_64 / name)
                class_counts[label] += 1
                saved += 1
                pbar.update(1)

pbar.close()
print("Done! Saved", saved, "images. Per class:", class_counts)