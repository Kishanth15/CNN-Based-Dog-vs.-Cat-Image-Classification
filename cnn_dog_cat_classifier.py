"""CNN dog-vs-cat image classifier using MobileNetV2 transfer learning.

Expected folders: DATA_DIR/{train,validation,test}/{cat,dog}/image.jpg

Example:
    python cnn_dog_cat_classifier.py --data-dir ./data

The first run downloads MobileNetV2 ImageNet weights. The script evaluates
only on the held-out test folder, prints classification metrics, and saves a
plot and trained Keras model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["cat", "dog"]
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def load_split(directory: Path, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    """Load a labeled split with a consistent cat=0, dog=1 label mapping."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset split folder not found: {directory}")
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=SEED,
    )
    return dataset.prefetch(AUTOTUNE)


def build_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build an augmented MobileNetV2 transfer-learning classifier."""
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1, name="mobilenet_scaling")(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="dog_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="dog_cat_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    return model, backbone


def combine_histories(*histories: tf.keras.callbacks.History) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(values)
    return merged


def save_evaluation_plot(
    history: dict[str, list[float]], y_true: np.ndarray, y_prob: np.ndarray,
    y_pred: np.ndarray, output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    fig.suptitle("CNN Dog vs. Cat — Held-out Test Evaluation", fontsize=17, weight="bold")

    ax = axes[0, 0]
    ax.plot(history.get("accuracy", []), label="Training accuracy", color="#3478c9")
    ax.plot(history.get("val_accuracy", []), label="Validation accuracy", color="#e36a5c")
    ax.set(title="Accuracy by epoch", xlabel="Epoch", ylabel="Accuracy")
    ax.legend()
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    ax.plot(history.get("loss", []), label="Training loss", color="#3478c9")
    ax.plot(history.get("val_loss", []), label="Validation loss", color="#e36a5c")
    ax.set(title="Binary cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    ax.legend()
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=[0, 1], display_labels=CLASS_NAMES,
        ax=ax, cmap="Blues", colorbar=False, values_format="d",
    )
    ax.set_title("Test-set confusion matrix")

    ax = axes[1, 1]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, color="#3478c9", linewidth=2.5, label=f"CNN (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="No-skill baseline")
    ax.set(title="Receiver operating characteristic", xlabel="False positive rate",
           ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder containing train/, validation/, and test/ splits.")
    parser.add_argument("--epochs", type=int, default=12,
                        help="Maximum epochs for each training phase (default: 12).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    train_ds = load_split(args.data_dir / "train", args.batch_size, shuffle=True)
    val_ds = load_split(args.data_dir / "validation", args.batch_size, shuffle=False)
    test_ds = load_split(args.data_dir / "test", args.batch_size, shuffle=False)

    model, backbone = build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                             patience=2, min_lr=1e-6),
    ]

    print("Phase 1: train the new classification head")
    head_history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                             callbacks=callbacks, verbose=1)

    # Fine-tune only the final backbone layers; keep batch-normalization frozen.
    print("\nPhase 2: fine-tune the final MobileNetV2 layers")
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    fine_history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                             callbacks=callbacks, verbose=1)

    y_true = np.concatenate([labels.numpy().reshape(-1) for _, labels in test_ds]).astype(int)
    y_prob = model.predict(test_ds, verbose=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {
        "Accuracy": accuracy,
        "Precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 score (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }
    print("\nHeld-out test metrics (threshold = 0.50)")
    for name, value in metrics.items():
        print(f"{name:20s}: {value:.4f}")
    print("\nClassification report")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=CLASS_NAMES, digits=4, zero_division=0))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_evaluation_plot(combine_histories(head_history, fine_history), y_true, y_prob,
                         y_pred, output_dir / "dog_cat_cnn_results.png")
    model.save(output_dir / "dog_cat_cnn.keras")
    print(f"Evaluation graph: {output_dir / 'dog_cat_cnn_results.png'}")
    print(f"Saved model: {output_dir / 'dog_cat_cnn.keras'}")
    if accuracy < 0.85:
        raise RuntimeError(
            f"Test accuracy {accuracy:.1%} is below the requested 85% target. "
            "Check dataset quality, labels, and class balance."
        )
    print("Accuracy target: PASSED (at least 85% on the held-out test set).")


if __name__ == "__main__":
    main()
