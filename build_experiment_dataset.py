"""
Build the imbalanced experiment dataset from your existing `images/` folder.

Expects:
    images/
        aircraft/   (10,000 images)
        bird/       (10,000 images)
        butterfly/  (10,000 images)
        car/        (10,000 images)
        cat/        (10,000 images)
        dog/        (10,000 images)
        fish/       (10,000 images)
        flower/     (10,000 images)
        human/      (10,000 images)
        ship/       (10,000 images)

Classes are sorted alphabetically:
    aircraft, bird, butterfly, car, cat, dog, fish, flower, human, ship

First 5 alphabetically -> HIGH density: take every 10th image -> 1000 images
Last 5 alphabetically  -> LOW density:  take every 50th image -> 200 images

Output goes to a new folder, one subfolder per class, named so that sorting
them alphabetically reproduces class indices 0-9 in the same high/low order:

    output/
        class_00_aircraft_high/   1000 images
        class_01_bird_high/       1000 images
        class_02_butterfly_high/  1000 images
        class_03_car_high/        1000 images
        class_04_cat_high/        1000 images
        class_05_dog_low/         200 images
        class_06_fish_low/        200 images
        class_07_flower_low/      200 images
        class_08_human_low/       200 images
        class_09_ship_low/        200 images

This output folder is ready to pass straight to dataset.py / train.py as
--data_root.
"""
import os
import argparse
import shutil

IMG_EXTS = (".jpg", ".jpeg", ".png")


def list_images(class_dir):
    files = [f for f in os.listdir(class_dir) if f.lower().endswith(IMG_EXTS)]
    files.sort()  # deterministic order so "every Nth" is reproducible
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_root", required=True, help="path to your existing 'images' folder")
    parser.add_argument("--output_dir", default="data", help="where to build the experiment dataset")
    parser.add_argument("--high_stride", type=int, default=10, help="take every Nth image for high-density classes")
    parser.add_argument("--low_stride", type=int, default=50, help="take every Nth image for low-density classes")
    parser.add_argument("--high_target", type=int, default=1000)
    parser.add_argument("--low_target", type=int, default=200)
    parser.add_argument("--symlink", action="store_true",
                         help="symlink instead of copy (faster, saves disk - use on the server)")
    args = parser.parse_args()

    all_classes = sorted(
        d for d in os.listdir(args.images_root)
        if os.path.isdir(os.path.join(args.images_root, d))
    )
    if len(all_classes) != 10:
        print(f"WARNING: expected 10 class folders, found {len(all_classes)}: {all_classes}")

    high_classes = all_classes[:5]
    low_classes = all_classes[5:]
    print("high-density classes (1000 each):", high_classes)
    print("low-density classes  (100 each):", low_classes)

    os.makedirs(args.output_dir, exist_ok=True)

    for idx, cls in enumerate(all_classes):
        is_high = cls in high_classes
        stride = args.high_stride if is_high else args.low_stride
        target = args.high_target if is_high else args.low_target
        tag = "high" if is_high else "low"

        src_dir = os.path.join(args.images_root, cls)
        files = list_images(src_dir)

        if len(files) < stride * target:
            print(f"  WARNING: '{cls}' has {len(files)} images, "
                  f"needs at least {stride * target} to take every {stride}th "
                  f"up to {target}. Will take as many as available on that stride.")

        selected = files[::stride][:target]

        out_dir = os.path.join(args.output_dir, f"class_{idx:02d}_{cls}_{tag}")
        os.makedirs(out_dir, exist_ok=True)

        for fname in selected:
            src = os.path.join(src_dir, fname)
            dst = os.path.join(out_dir, fname)
            if args.symlink:
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(src), dst)
            else:
                shutil.copy2(src, dst)

        print(f"class {idx} ({cls}, {tag}): selected {len(selected)} images -> {out_dir}")

    print("\ndone. pass this folder as --data_root to train.py:")
    print(f"  python train.py --data_root {args.output_dir}")


if __name__ == "__main__":
    main()
