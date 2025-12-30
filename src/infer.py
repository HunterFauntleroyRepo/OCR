# src/infer_tf.py
import os
import cv2
import numpy as np
import tensorflow as tf
from charset import idx_to_char, NUM_CLASSES

# CTC blank index
CTC_BLANK = NUM_CLASSES - 1 # last index for blank

# Paths
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "ocr_simple_tf.keras"))

IMG_HEIGHT = 32
IMG_WIDTH = 128

def load_model(model_dir=MODEL_DIR):
    """
    Load a TensorFlow SavedModel for OCR inference.
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model folder not found: {model_dir}")
    model = tf.keras.models.load_model(model_dir)
    return model

def preprocess_image_tf(img_path):
    """
    Preprocess an image for TF CRNN inference.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)        # [H,W,1]
    img = np.expand_dims(img, axis=0)         # [1,H,W,1] batch dim
    return img

def greedy_decode_tf(preds):
    """
    Greedy CTC decoding for TF output [B, W, C].
    """
    preds = np.argmax(preds, axis=-1)  # [B, W]
    texts = []
    for seq in preds:
        text = ""
        prev = -1
        for p in seq:
            if p != prev and p != CTC_BLANK:
                text += idx_to_char[p]
            prev = p
        texts.append(text)
    return texts

def run_inference():
    model = load_model()

    img_path = input("Enter path to image: ").strip()
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = preprocess_image_tf(img_path)
    preds = model.predict(img)  # [1, W, C]
    text = greedy_decode_tf(preds)
    print("Predicted text:", text[0])

if __name__ == "__main__":
    run_inference()
