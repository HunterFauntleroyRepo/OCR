# src/infer.py
import torch
import cv2
from src.model import CRNN
from src.charset import idx_to_char, NUM_CLASSES
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    """
    Load a pre-trained CRNN model from disk and prepare it for inference.
    This function initializes a CRNN model with the specified number of classes,
    loads pre-trained weights from the given model path, and sets the model to
    evaluation mode for inference.
    Args:
        model_path (str): Path to the saved model state dictionary file.
    Returns:
        torch.nn.Module: A CRNN model in evaluation mode with loaded weights,
                        moved to the specified device (CPU or GPU).
    Raises:
        FileNotFoundError: If the model file at model_path does not exist.
        RuntimeError: If the model architecture doesn't match the saved state dict.
    """

    model = CRNN(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def preprocess_image(img_path):
    """
    Preprocesses an image for OCR inference.
    Reads an image from the specified file path, converts it to grayscale,
    resizes it to a standard dimension, normalizes pixel values, and converts
    it to a PyTorch tensor on the appropriate device.
    Args:
        img_path (str): File path to the input image.
    Returns:
        torch.Tensor: Preprocessed image tensor of shape [1, 1, H, W] on the
                     configured device (CPU or GPU). Values are normalized to [0, 1].
    Note:
        - Image is converted to grayscale (single channel)
        - Image is resized to 128x32 pixels
        - Pixel values are normalized by dividing by 255.0
        - Tensor includes batch and channel dimensions for model input
    """

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 32))
    img = img.astype("float32") / 255.0
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return img.to(DEVICE)

def greedy_decode(preds):
    """
    Decode model predictions using greedy decoding strategy.
    Converts raw model logits to text by selecting the highest probability class
    at each time step, then removes consecutive duplicates and blank tokens.
    Args:
        preds (torch.Tensor): Model prediction logits of shape [B, W, NUM_CLASSES]
            where B is batch size, W is sequence width, and NUM_CLASSES is the
            number of character classes.
    Returns:
        list: List of decoded text strings, one for each item in the batch.
    Notes:
        - Assumes blank token is at index NUM_CLASSES-1
        - Removes consecutive duplicate predictions (CTC-style decoding)
        - Filters out blank/padding tokens from output
        - Uses idx_to_char mapping to convert indices to characters
    """

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
    """
    Executes the OCR inference pipeline to predict text from an image.
    This function performs the following steps:
    1. Locates and loads a pre-trained CRNN OCR model from the models directory
    2. Prompts the user to input an image file path
    3. Preprocesses the input image
    4. Runs inference on the preprocessed image using the loaded model
    5. Decodes the model predictions into readable text using greedy decoding
    6. Prints the predicted text to the console
    Raises:
        Exception: If the trained model file is not found at the expected location.
        Exception: If the provided image file path does not exist.
    Returns:
        None: The function prints the predicted text to stdout instead of returning a value.
    Note:
        - Requires a trained model file (crnn_ocr.pth) to be present in the models directory
        - Uses torch.no_grad() context to disable gradient computation for inference efficiency
        - The model output shape is expected to be [B, W, num_classes] where B is batch size,
          W is sequence width, and num_classes is the number of character classes
    """

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
