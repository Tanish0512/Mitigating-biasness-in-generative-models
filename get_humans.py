import io
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm.std import tqdm

REPO = "huggan/CelebA-faces-with-attributes"
TOTAL = 10_000
OUT_128 = Path("images/human_128")
OUT_64 = Path("images/human_64")
OUT_128.mkdir(parents=True, exist_ok=True)
OUT_64.mkdir(parents=True, exist_ok=True)


def center_crop_square(img):
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


saved = 0
pbar = tqdm(total=TOTAL, desc="Saving human faces")

for shard_idx in range(132):
    if saved >= TOTAL:
        break
    path = hf_hub_download(
        repo_id=REPO, repo_type="dataset",
        filename=f"data/train-{shard_idx:05d}-of-00132.parquet",
        token=os.getenv("HF_TOKEN"),
    )
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=500, columns=["image"]):
        for item in batch.column("image").to_pylist():
            if saved >= TOTAL:
                break
            img = Image.open(io.BytesIO(item["bytes"])).convert("RGB")
            img = center_crop_square(img).resize((128, 128), Image.LANCZOS)
            name = f"{saved:05d}.png"
            img.save(OUT_128 / name)
            img.resize((64, 64), Image.LANCZOS).save(OUT_64 / name)
            saved += 1
            pbar.update(1)

pbar.close()
print("Done! Saved", saved, "face images")