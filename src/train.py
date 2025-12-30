import os
import tensorflow as tf
from dataset import OCRDataset
from model import build_crnn

# Paths
TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "train"))
VAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "val"))

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 95

# Datasets
train_dataset = OCRDataset(TRAIN_DIR).get_dataset(batch_size=BATCH_SIZE)
val_dataset = OCRDataset(VAL_DIR).get_dataset(batch_size=BATCH_SIZE, shuffle=False)

# Model
model = build_crnn(num_classes=NUM_CLASSES)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# CTC Loss function
def ctc_loss(y_true, y_pred, label_length):
    """
    y_true: padded label sequences [B, max_label_len]
    y_pred: [B, W, num_classes]
    label_length: actual length of each sequence [B,1]
    """
    batch_len = tf.shape(y_true)[0]
    input_len = tf.shape(y_pred)[1]
    input_len = input_len * tf.ones(shape=(batch_len,1), dtype=tf.int32)
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_len, label_length)
    return loss


# Training step
@tf.function
def train_step(images, labels, label_length):
    with tf.GradientTape() as tape:
        y_pred = model(images, training=True)
        loss = ctc_loss(labels, y_pred, label_length)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Training loop
for epoch in range(1, EPOCHS+1):
    print(f"Epoch {epoch}/{EPOCHS}")
    running_loss = 0
    num_batches = 0
    for batch in train_dataset:
        loss = train_step(batch['image'], batch['label'], batch['label_length'])
        running_loss += tf.reduce_mean(loss)
        num_batches += 1

    print(f"Training loss: {running_loss/num_batches:.4f}")


# Save the model
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
model_path = os.path.join(models_dir, "ocr_simple_tf.keras")
model.save(model_path)
print(f"Model saved to {model_path}")
print("Training complete.")
