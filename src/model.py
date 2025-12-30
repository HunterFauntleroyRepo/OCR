# src/model.py
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models

def build_crnn(input_shape=(32,128,1), num_classes=95):
    inputs = layers.Input(shape=input_shape, name="image")

    # CNN
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Collapse height dimension
    x = layers.Reshape((x.shape[2], x.shape[1]*x.shape[3]))(x)  # (W, features)

    # RNN
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model
