import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ClassImageDataset(Dataset):
    """
    Expects a folder structure like:

        data_root/
            class_00_high/
            class_01_high/
            class_02_high/
            class_03_high/
            class_04_high/
            class_05_low/
            class_06_low/
            class_07_low/
            class_08_low/
            class_09_low/
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
                    self.samples.append(
                        (os.path.join(cls_dir, fname), label)
                    )

        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.LANCZOS
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
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
    import sys

    ds = ClassImageDataset(sys.argv[1])

    print("classes:", ds.classes)
    print("counts:", ds.class_counts())
    print("total images:", len(ds))
