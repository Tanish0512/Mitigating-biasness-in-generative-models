import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ClassImageDataset(Dataset):
    """
    Expects a folder structure like:

        data_root/
            class_00_high/   -> 1000 images (high-density class)
            class_01_high/   -> 1000 images
            class_02_high/   -> 1000 images
            class_03_high/   -> 1000 images
            class_04_high/   -> 1000 images
            class_05_low/    -> 100 images  (low-density class)
            class_06_low/    -> 100 images
            class_07_low/    -> 100 images
            class_08_low/    -> 100 images
            class_09_low/    -> 100 images

    Folder names are sorted alphabetically to assign class indices 0..N-1.
    Name them so "high" classes and "low" classes are easy to tell apart
    later when you read the results (the prefix in the example above works).
    """

    def __init__(self, data_root, image_size=64):
        self.classes = sorted(
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        )
        self.samples = []
        for label, cls in enumerate(self.classes):
            cls_dir = os.path.join(data_root, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), label))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                               interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1], DDPM convention
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

    def class_counts(self):
        counts = {c: 0 for c in self.classes}
        for _, label in self.samples:
            counts[self.classes[label]] += 1
        return counts


if __name__ == "__main__":
    # quick sanity check: python dataset.py /path/to/data_root
    import sys
    ds = ClassImageDataset(sys.argv[1])
    print("classes:", ds.classes)
    print("counts:", ds.class_counts())
    print("total images:", len(ds))
