# src/generate_data.py
# Generate synthetic ASCII text images for OCR training

import os
import random
from PIL import Image, ImageDraw, ImageFont
from charset import ASCII_CHARS

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
MIN_LEN = 5
MAX_LEN = 10

def get_fonts() -> list:
    """
    Retrieves all available font files from the fonts directory.
    This method scans the FONT_DIR directory and collects all font files
    with .ttf (TrueType) or .otf (OpenType) extensions. These fonts are
    typically used for text rendering in data generation or image processing tasks.
    Returns:
        list: A list of absolute file paths to all valid font files found
              in the FONT_DIR directory.
    Raises:
        Exception: If no font files (.ttf or .otf) are found in the FONT_DIR
                  directory, an exception is raised to prevent downstream
                  operations from failing due to missing font resources.
    """

    fonts = []
    for f in os.listdir(FONT_DIR):
        if f.endswith(".ttf") or f.endswith(".otf"):
            fonts.append(os.path.join(FONT_DIR, f))
    if not fonts:
        raise Exception("No fonts found in folder")
    return fonts

def generate_image(text, font_path):
    """
    Generate a grayscale image with centered text.
    This function creates a new image with a white background and draws the provided
    text in black, centered vertically on the image. It attempts to use a TrueType font
    from the specified path, falling back to the default font if unavailable.
    Args:
        text (str): The text string to render on the image.
        font_path (str): The file path to a TrueType font file (.ttf).
    Returns:
        PIL.Image.Image: A grayscale image (mode "L") with the rendered text,
                         with dimensions (IMG_WIDTH, IMG_HEIGHT).
    """

    img = Image.new("L", (IMG_WIDTH, IMG_HEIGHT), color=255)  # white background
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, 24)
    except:
        font = ImageFont.load_default()

    # Center text vertically
    bbox = draw.textbbox((0, 0), text, font=font)
    height = bbox[3] - bbox[1] # bottom - top
    y = (IMG_HEIGHT - height) // 2

    # Draw text
    draw.text((5, y), text, font=font, fill=0)  # black text
    return img

def generate_dataset(output_dir, num_samples):
    """
    Generate a dataset of synthetic OCR training images with random text.
    This function creates a specified number of PNG images containing randomly generated
    text rendered in various fonts. The images are saved to the output directory with
    sanitized filenames based on the text content. Progress is printed every 100 images.
    Args:
        output_dir (str): The directory path where generated images will be saved.
            The directory will be created if it does not exist.
        num_samples (int): The number of synthetic images to generate.
    Returns:
        None
    """

    fonts = get_fonts()
    for i in range(num_samples):
        length = random.randint(MIN_LEN, MAX_LEN)
        text = "".join(random.choices(ASCII_CHARS, k=length))
        font_path = random.choice(fonts)
        img = generate_image(text, font_path)

        filename = f"{i}_{text}.png"
        img.save(os.path.join(output_dir, filename))
        if i % 100 == 0:
            print(f"Generated {i}/{num_samples} images in {output_dir}")

if __name__ == "__main__":
    print("Generating training data...")
    generate_dataset(TRAIN_DIR, NUM_TRAIN)
    print("Generating validation data...")
    generate_dataset(VAL_DIR, NUM_VAL)
    print("Done!")
