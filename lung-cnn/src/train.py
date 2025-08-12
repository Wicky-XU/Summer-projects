# train.py
# High-level training loop with callbacks and saving artifacts.

import os
import tensorflow as tf
from .data import set_seeds, build_datasets
from .model import build_model, compile_model

def train(
    data_dir: str,
    img_size=(224, 224),
    batch_size: int = 32,
    lr: float = 1e-3,
    epochs: int = 50,
    val_split: float = 0.2,
    seed: int = 50,
    output_dir: str = "results",
    model_path: str = "models/best.keras",
    use_augmentation: bool = True,
):
    """Train a CNN on images in `data_dir` and save best model & logs."""
    set_seeds(seed)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    train_ds, val_ds, class_names = build_datasets(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        val_split=val_split,
        seed=seed,
    )

    model = build_model(num_classes=len(class_names),
                        img_size=img_size + (3,),
                        use_augmentation=use_augmentation)
    compile_model(model, lr=lr)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(output_dir, "history.csv")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on val as a quick report
    val_loss, val_acc, val_auc = model.evaluate(val_ds, verbose=0)
    print(f"Val acc={val_acc:.4f}, AUC={val_auc:.4f}")

    return model, history, class_names

if __name__ == "__main__":
    # Example usage (adjust data_dir to your local path)
    train(data_dir="data", epochs=10)
