"""
Rough, dependency-light diagnostic to compare generated output quality across
classes. This is NOT your HDAI metric (I don't have that formula) - use this
to get a quick number for the slide, then plug the real generated samples
into your own HDAI computation for the final figure.

For each class folder of generated samples, reports:
  - diversity: mean pairwise pixel difference (low diversity ~ mode collapse,
    which is what you'd expect to see more of in your low-density classes)
  - sharpness: mean Laplacian variance (rough proxy for "well-formed" vs
    "blurry/noisy" output)
"""
import os
import argparse
import numpy as np
from PIL import Image
import cv2


def load_images(folder):
    imgs = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(".png"):
            img = np.array(Image.open(os.path.join(folder, fname)).convert("L"))
            imgs.append(img)
    return imgs


def diversity_score(imgs, max_pairs=30):
    if len(imgs) < 2:
        return 0.0
    arr = np.stack(imgs[:max_pairs]).astype(np.float32)
    diffs = [
        np.mean(np.abs(arr[i] - arr[j]))
        for i in range(len(arr)) for j in range(i + 1, len(arr))
    ]
    return float(np.mean(diffs)) if diffs else 0.0


def sharpness_score(imgs):
    if not imgs:
        return 0.0
    scores = [cv2.Laplacian(img, cv2.CV_64F).var() for img in imgs]
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", required=True, help="output_dir passed to sample.py")
    args = parser.parse_args()

    print(f"{'class':20s} {'n_imgs':>8s} {'diversity':>12s} {'sharpness':>12s}")
    rows = []
    for cls_name in sorted(os.listdir(args.samples_dir)):
        cls_dir = os.path.join(args.samples_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        imgs = load_images(cls_dir)
        div = diversity_score(imgs)
        sharp = sharpness_score(imgs)
        rows.append((cls_name, len(imgs), div, sharp))
        print(f"{cls_name:20s} {len(imgs):8d} {div:12.2f} {sharp:12.2f}")

    # quick summary line worth putting straight on the slide
    if rows:
        avg_div = np.mean([r[2] for r in rows])
        print(f"\naverage diversity across all classes: {avg_div:.2f}")
        print("compare the 'high' vs 'low' rows above directly -- a noticeably")
        print("lower diversity or sharpness score on your low-density classes")
        print("is your first quantitative evidence of bias for the slide.")


if __name__ == "__main__":
    main()
