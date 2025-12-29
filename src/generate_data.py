# src/generate_data.py
# Generate synthetic ASCII text images for OCR training

import os
import random
from PIL import Image, ImageDraw, ImageFont
import string
from src.charset import ASCII_CHARS
import re

# Paths
TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "train"))
VAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "val"))
FONT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fonts"))


# Image size
IMG_WIDTH = 128
IMG_HEIGHT = 32

# Number of samples
NUM_TRAIN = 2000   # you can increase later
NUM_VAL = 400

# Text length range
MIN_LEN = 3
MAX_LEN = 10

def get_fonts():
    fonts = []
    for f in os.listdir(FONT_DIR):
        if f.endswith(".ttf") or f.endswith(".otf"):
            fonts.append(os.path.join(FONT_DIR, f))
    if not fonts:
        raise Exception("No fonts found in fonts/ folder!")
    return fonts

def generate_image(text, font_path):
    img = Image.new("L", (IMG_WIDTH, IMG_HEIGHT), color=255)  # white background
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, 24)
    except:
        font = ImageFont.load_default()

    # Center text vertically
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # fallback for older Pillow versions
        w, h = font.getsize(text)

    y = (IMG_HEIGHT - h) // 2
    draw.text((5, y), text, font=font, fill=0)  # black text
    return img

def generate_dataset(output_dir, num_samples):
    os.makedirs(output_dir, exist_ok=True)
    fonts = get_fonts()
    for i in range(num_samples):
        length = random.randint(MIN_LEN, MAX_LEN)
        text = "".join(random.choices(ASCII_CHARS, k=length))
        font_path = random.choice(fonts)
        img = generate_image(text, font_path)

        # sanitize label for use in filenames
        safe_text = re.sub(r'[<>:"/\\|?*\n\r\t]', '_', text).strip().rstrip('. ')
        safe_text = safe_text[:100]  # optional length limit

        os.makedirs(output_dir, exist_ok=True)
        filename = f"{i}_{safe_text}.png"
        img.save(os.path.join(output_dir, filename))
        if i % 100 == 0:
            print(f"Generated {i}/{num_samples} images in {output_dir}")

if __name__ == "__main__":
    print("Generating training data...")
    generate_dataset(TRAIN_DIR, NUM_TRAIN)
    print("Generating validation data...")
    generate_dataset(VAL_DIR, NUM_VAL)
    print("Done!")
