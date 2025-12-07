import os
import json
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

def fit_model(model, train_ds, val_ds, epochs=30, out_dir="experiments/dl_checkpoints", name="model", patience=6):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{name}.keras")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    # Save history JSON
    hist_path = os.path.join(out_dir, f"{name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history.history, f)

    # Create figures
    fig_acc, fig_loss = None, None
    try:
        # Accuracy plot
        fig_acc, ax_acc = plt.subplots(figsize=(8,4))
        acc = history.history.get("accuracy", history.history.get("acc", []))
        val_acc = history.history.get("val_accuracy", history.history.get("val_acc", []))
        ax_acc.plot(acc, label="train_acc")
        ax_acc.plot(val_acc, label="val_acc")
        ax_acc.set_title("Accuracy")
        ax_acc.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_acc.png"))

        # Loss plot
        fig_loss, ax_loss = plt.subplots(figsize=(8,4))
        loss = history.history.get("loss", [])
        val_loss = history.history.get("val_loss", [])
        ax_loss.plot(loss, label="train_loss")
        ax_loss.plot(val_loss, label="val_loss")
        ax_loss.set_title("Loss")
        ax_loss.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_loss.png"))
    except Exception as e:
        print("Warning: could not generate plots.", e)

    return history, fig_acc, fig_loss