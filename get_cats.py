import io
import math
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm.std import tqdm

REPO = "benjamin-paine/imagenet-1k-128x128"


def fetch_cats_sharded(
    out_128: str = "images/cat_128",
    out_64: str = "images/cat_64",
    total_target: int = 6500,
    start_label: int = 281,   # tabby
    end_label: int = 285,     # Egyptian cat
    hf_token: str | None = None,
):
    p128, p64 = Path(out_128), Path(out_64)
    p128.mkdir(parents=True, exist_ok=True)
    p64.mkdir(parents=True, exist_ok=True)

    num_classes = end_label - start_label + 1
    max_per_class = math.ceil(total_target / num_classes)
    class_counts = {i: 0 for i in range(start_label, end_label + 1)}

    total_saved = 0
    pbar = tqdm(total=total_target, desc="Saving cat images")
    num_train_shards = 13

    for shard_idx in range(num_train_shards):
        if total_saved >= total_target:
            break

        shard_file = f"data/train-{shard_idx:05d}-of-{num_train_shards:05d}.parquet"
        print(f"\n[Shard {shard_idx + 1}/{num_train_shards}] {shard_file}")
        path = hf_hub_download(
            repo_id=REPO, repo_type="dataset", filename=shard_file, token=hf_token
        )

        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1000, columns=["label", "image"]):
            labels = batch.column("label").to_pylist()
            images = batch.column("image")

            for i, label in enumerate(labels):
                if total_saved >= total_target:
                    break
                if label in class_counts and class_counts[label] < max_per_class:
                    img_bytes = images[i]["bytes"].as_py()
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    if img.size != (128, 128):
                        img = img.resize((128, 128), Image.LANCZOS)

                    name = f"{total_saved:05d}.png"
                    img.save(p128 / name)
                    img.resize((64, 64), Image.LANCZOS).save(p64 / name)

                    class_counts[label] += 1
                    total_saved += 1
                    pbar.update(1)

    pbar.close()
    print(f"Done! Saved {total_saved} cats. Per class: {class_counts}")


if __name__ == "__main__":
    fetch_cats_sharded(hf_token=os.getenv("HF_TOKEN"))