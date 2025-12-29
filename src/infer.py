# src/infer.py
import torch
import cv2
from src.model import CRNN
from src.charset import idx_to_char, NUM_CLASSES
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = CRNN(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 32))
    img = img.astype("float32") / 255.0
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return img.to(DEVICE)

def greedy_decode(preds):
    preds = preds.argmax(2)  # [B,W]
    text_list = []
    for i in range(preds.size(0)):
        pred_text = ""
        prev = -1
        for p in preds[i]:
            p = p.item()
            if p != prev and p != NUM_CLASSES-1:
                pred_text += idx_to_char[p]
            prev = p
        text_list.append(pred_text)
    return text_list

def run_inference():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "crnn_ocr.pth"))
    if not os.path.exists(model_path):
        raise Exception("Trained model not found. Train the model first using train.py")

    model = load_model(model_path)

    img_path = input("Enter path to image: ").strip()
    if not os.path.exists(img_path):
        raise Exception("Image not found!")

    img = preprocess_image(img_path)
    with torch.no_grad():
        preds = model(img)          # [B, W, num_classes]
    text = greedy_decode(preds)
    print("Predicted text:", text[0])

if __name__ == "__main__":
    run_inference()
