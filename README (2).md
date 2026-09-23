# Small-scale DDPM bias experiment (5 classes x 1000 imgs, 5 classes x 200 imgs)

## 1. Install dependencies
```
pip install -r requirements.txt
```

## 2. Build the experiment dataset from your existing `images/` folder
```
python build_experiment_dataset.py --images_root images/ --output_dir data/ --symlink
```
Your source folder should have one subfolder per class with ~10,000 images
each (`aircraft, bird, butterfly, car, cat, dog, fish, flower, human, ship`).
Sorted alphabetically, the first 5 classes become high-density (every 10th
image -> 1000 images) and the last 5 become low-density (every 50th image ->
200 images). This writes:

```
data/
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
```
Use `--symlink` on the server (instant, no extra disk); drop it for real
file copies if you want a standalone copy of `data/`.

## 3. Sanity-check the dataset loads correctly (do this locally in VS Code first)
```
python dataset.py data/
```
This just prints the classes found and per-class counts. Fix this before
touching the server - it catches path/naming mistakes for free, with no GPU
needed.

## 4. Train (do this on the server/GPU)
```
python train.py --data_root data/ --num_steps 6000 --batch_size 32
```
- `--num_steps 6000` is a reasonable proof-of-concept budget (not
  photorealistic, but enough to show class-specific structure).
- Checkpoints are saved to `checkpoints/unet_stepN.pt` every
  `--save_every` steps (default 1000), so you can sample from an
  earlier checkpoint if you're short on time.

## 5. Sample from the trained model
```
python sample.py --checkpoint checkpoints/unet_step6000.pt --num_per_class 50
```
Generates 50 images per class into `samples/<class_name>/`.

## 6. Run the diagnostic
```
python bias_diagnostic.py --samples_dir samples/
```
Prints a diversity + sharpness score per class. Compare the "high" rows
against the "low" rows - a visible gap is your first quantitative bias
evidence for the slide. Feed the same generated samples into your actual
HDAI formula for the number you'll headline with.

## Time budget (from GPU model)
| Hardware | Train (6000 steps) | Sample (50/class = 500 imgs, DDIM-50) | Total |
|---|---|---|---|
| CPU only | 12-24+ hrs | 20-40 min | avoid |
| Colab free T4 | ~20-30 min | ~3-6 min | ~30-40 min |
| RTX 3060/4060 | ~15-25 min | ~2-5 min | ~20-30 min |
| RTX 3090/4090/A100 | ~6-12 min | ~1-3 min | ~10-15 min |

Once you check `nvidia-smi` on your server, tell me the GPU model and I'll
tighten this estimate.

## Server workflow reminder
Run training inside `tmux` (`tmux new -s training`) so an SSH drop doesn't
kill the job:
```
tmux new -s training
python train.py --data_root data/ --num_steps 6000
# ctrl+b, d to detach; tmux attach -t training to come back
```
