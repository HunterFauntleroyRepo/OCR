# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import OCRDataset
from src.model import CRNN
from src.charset import idx_to_char, NUM_CLASSES
from tqdm import tqdm
import os

# Paths
TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "train"))
VAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "val"))

# Hyperparameters
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_dataset = OCRDataset(TRAIN_DIR)
val_dataset = OCRDataset(VAL_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# Initialize model, optimizer, loss
model = CRNN(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CTCLoss(blank=NUM_CLASSES-1, zero_infinity=True)  # last index is used as blank

def train_one_epoch(epoch):
    model.train()
    running_loss = 0
    for imgs, labels, lengths in tqdm(train_loader, desc=f"Epoch {epoch}"):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        preds = model(imgs)                  # [B, W, num_classes]
        preds = preds.permute(1,0,2)         # CTC expects [W, B, C]
        pred_lengths = torch.full(size=(preds.size(1),), fill_value=preds.size(0), dtype=torch.long)

        loss = criterion(preds, labels, pred_lengths, lengths)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch} training loss: {running_loss/len(train_loader):.4f}")

def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels, lengths in val_loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs)
            preds = preds.argmax(2)           # greedy decode [B,W]
            for i in range(preds.size(0)):
                pred_text = ""
                prev = -1
                for p in preds[i]:
                    p = p.item()
                    if p != prev and p != NUM_CLASSES-1:
                        pred_text += idx_to_char[p]
                    prev = p
                label_text = "".join([idx_to_char[l.item()] for l in labels[i]])
                if pred_text == label_text:
                    correct +=1
                total +=1
    acc = correct / total
    print(f"Validation Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    for epoch in range(1, EPOCHS+1):
        train_one_epoch(epoch)
        evaluate()

    # Save the trained model
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "crnn_ocr.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
