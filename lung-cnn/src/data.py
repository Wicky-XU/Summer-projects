# data.py
# Utilities to set seeds and build tf.data pipelines for image classification.

import os, random
import numpy as np
import tensorflow as tf

def set_seeds(seed: int = 50) -> None:
    """Set all relevant random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_augmentation() -> tf.keras.Sequential:
    """Return a light augmentation pipeline."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ], name="augmentation")

def build_datasets(
    data_dir: str,
    img_size=(224, 224),
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 50,
):
    """
    Build training/validation datasets from a single directory structure:
    data_dir/
      class_a/
      class_b/
      ...
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )
    class_names = train_ds.class_names

    # Normalize to [0,1] and speed up with AUTOTUNE
    norm = tf.keras.layers.Rescaling(1.0 / 255.0)
    AUTOTUNE = tf.data.AUTOTUNE

    def _prep(ds):
        ds = ds.map(lambda x, y: (norm(x), y), num_parallel_calls=AUTOTUNE)
        return ds.cache().prefetch(AUTOTUNE)

    return _prep(train_ds), _prep(val_ds), class_names
