from PIL import Image


def resize_image(image: Image.Image, size: tuple[int, int] = (64, 64)) -> Image.Image:
    """Return an RGB image resized to the requested dimensions."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.resize(size, Image.Resampling.LANCZOS)
