# src/dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset
from src.charset import char_to_idx, NUM_CLASSES

class OCRDataset(Dataset):
    def __init__(self, folder):
        self.files = []
        for f in os.listdir(folder):
            if f.endswith(".png"):
                # filename format: 0_TEXT.png
                label = f.split("_")[1].split(".")[0]
                self.files.append((os.path.join(folder, f), label))

    def __len__(self):
        return len(self.files)

    def encode_label(self, text):
        return torch.LongTensor([char_to_idx[c] for c in text])

    def __getitem__(self, idx):
        img_path, label = self.files[idx]
        # read grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 32))
        img = img.astype("float32") / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # [1,H,W]

        label_tensor = self.encode_label(label)
        label_length = torch.tensor(len(label))
        return img, label_tensor, label_length
