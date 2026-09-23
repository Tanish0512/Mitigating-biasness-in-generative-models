import io
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image
from tqdm.std import tqdm

URL = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
ARCHIVE = Path("fgvc-aircraft-2013b.tar.gz")
TOTAL = 10_000            # poori 10,200 chahiye toh 10_200 kar do
BANNER = 20               # bottom pe copyright banner hota hai, use kaat dete hain
OUT_128 = Path("images/aircraft_128")
OUT_64 = Path("images/aircraft_64")
OUT_128.mkdir(parents=True, exist_ok=True)
OUT_64.mkdir(parents=True, exist_ok=True)


def center_crop_square(img):
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


# 1) Download (skip agar pehle se hai)
if not ARCHIVE.exists():
    part = Path(str(ARCHIVE) + ".part")
    with tqdm(unit="B", unit_scale=True, desc="Downloading") as bar:
        def hook(blocks, block_size, total):
            bar.total = total
            bar.update(blocks * block_size - bar.n)
        urllib.request.urlretrieve(URL, part, hook)
    part.rename(ARCHIVE)

# 2) Archive se images nikaalo, banner hatao, resize karo
saved = 0
pbar = tqdm(total=TOTAL, desc="Saving aircraft")
with tarfile.open(ARCHIVE, "r:gz") as tar:
    for m in tar:
        if saved >= TOTAL:
            break
        if not (m.isfile() and "/data/images/" in m.name and m.name.endswith(".jpg")):
            continue
        img = Image.open(io.BytesIO(tar.extractfile(m).read())).convert("RGB")
        w, h = img.size
        img = img.crop((0, 0, w, h - BANNER))
        img = center_crop_square(img).resize((128, 128), Image.LANCZOS)
        name = f"{saved:05d}.png"
        img.save(OUT_128 / name)
        img.resize((64, 64), Image.LANCZOS).save(OUT_64 / name)
        saved += 1
        pbar.update(1)

pbar.close()
print("Done! Saved", saved, "aircraft images")