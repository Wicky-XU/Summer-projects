# model.py
# Define a small CNN model (or switch to transfer learning later).

import tensorflow as tf

def build_model(num_classes: int, img_size=(224, 224, 3), use_augmentation=True):
    inputs = tf.keras.Input(shape=img_size, name="image")
    x = inputs
    if use_augmentation:
        from .data import get_augmentation
        x = get_augmentation()(x)

    # Simple CNN backbone
    def conv_block(x, f, k=3):
        x = tf.keras.layers.Conv2D(f, k, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        return x

    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="cnn_classifier")
    return model

def compile_model(model, lr: float = 1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model
