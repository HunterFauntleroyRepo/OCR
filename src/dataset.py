# src/dataset.py
import os
import cv2
import numpy as np
import tensorflow as tf
from charset import char_to_idx

class OCRDataset:
    def __init__(self, folder):
        """
        TensorFlow dataset for multi-character images.
        """
        self.files = []
        for f in os.listdir(folder):
            if f.endswith(".png"):
                label = f.split("_")[1].split(".")[0]  # full string label
                self.files.append((os.path.join(folder, f), label))

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 32))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)  # [32,128,1]
        return img

    def _encode_label(self, text):
        return np.array([char_to_idx[c] for c in text], dtype=np.int32)

    def generator(self):
        for path, label in self.files:
            img = self._load_image(path)
            label_seq = self._encode_label(label)
            label_len = np.array([len(label_seq)], dtype=np.int32)
            yield {"image": img, "label": label_seq, "label_length": label_len}

    def get_dataset(self, batch_size=32, shuffle=True):
        ds = tf.data.Dataset.from_generator(
            self.generator,
            output_signature={
                "image": tf.TensorSpec(shape=(32,128,1), dtype=tf.float32),
                "label": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                "label_length": tf.TensorSpec(shape=(1,), dtype=tf.int32)
            }
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=1000)
        ds = ds.padded_batch(batch_size,
            padded_shapes={
                "image": (32,128,1),
                "label": (None,),  # pad sequences
                "label_length": (1,)
            },
            padding_values={
                "image": 0.0,
                "label": 95,  # use 95 as blank padding
                "label_length": 0
            }
        )
        return ds
