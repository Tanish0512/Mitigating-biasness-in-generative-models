import os
import argparse
import torch
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler
from dataset import ClassImageDataset


def build_model(image_size, num_classes):
    return UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        num_class_embeds=num_classes,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="folder with one subfolder per class")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=6000, help="~5-8k is enough for a rough proof-of-concept")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=1000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = ClassImageDataset(args.data_root, image_size=args.image_size)
    print("classes:", dataset.classes)
    print("per-class counts:", dataset.class_counts())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=4, drop_last=True, pin_memory=(device == "cuda"))

    model = build_model(args.image_size, len(dataset.classes)).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    step = 0
    model.train()
    while step < args.num_steps:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (images.shape[0],), device=device
            ).long()
            noisy_images = scheduler.add_noise(images, noise, timesteps)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                noise_pred = model(noisy_images, timesteps, class_labels=labels).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            if step % 50 == 0:
                print(f"step {step}/{args.num_steps}  loss {loss.item():.4f}")

            if step % args.save_every == 0 or step == args.num_steps:
                ckpt_path = os.path.join(args.output_dir, f"unet_step{step}.pt")
                torch.save({
                    "model_state": model.state_dict(),
                    "classes": dataset.classes,
                    "image_size": args.image_size,
                }, ckpt_path)
                print("saved checkpoint:", ckpt_path)

            if step >= args.num_steps:
                break

    print("training done.")


if __name__ == "__main__":
    main()
