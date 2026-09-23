import os
import argparse
import torch
from diffusers import DDIMScheduler
from torchvision.utils import save_image
from train import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="samples")
    parser.add_argument("--num_per_class", type=int, default=50)
    parser.add_argument("--ddim_steps", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt["classes"]
    image_size = ckpt["image_size"]

    model = build_model(image_size, len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(args.ddim_steps)

    for label, cls_name in enumerate(classes):
        cls_dir = os.path.join(args.output_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        labels = torch.full((args.num_per_class,), label, device=device, dtype=torch.long)
        images = torch.randn(args.num_per_class, 3, image_size, image_size, device=device)

        for t in scheduler.timesteps:
            with torch.no_grad():
                noise_pred = model(images, t, class_labels=labels).sample
            images = scheduler.step(noise_pred, t, images).prev_sample

        images = (images.clamp(-1, 1) + 1) / 2  # back to [0, 1] for saving
        for i in range(args.num_per_class):
            save_image(images[i], os.path.join(cls_dir, f"sample_{i:03d}.png"))

        print(f"saved {args.num_per_class} samples for class '{cls_name}' -> {cls_dir}")


if __name__ == "__main__":
    main()
