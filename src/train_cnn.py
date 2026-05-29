import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
os_environ_set = __import__("os")
os_environ_set.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

DATA_PROC  = Path("data/processed")
MODELS_DIR = Path("models")
PLOTS_DIR  = Path("results/plots")
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 128

def load_data():
    X_train = np.load(DATA_PROC / "X_train.npy")
    X_test  = np.load(DATA_PROC / "X_test.npy")
    y_train = np.load(DATA_PROC / "y_train.npy")
    y_test  = np.load(DATA_PROC / "y_test.npy")

    # CNN needs shape (H, W, C)
    X_train = X_train[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    return X_train, X_test, y_train, y_test


def build_cnn():
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3,3), activation="relu",
                      input_shape=(IMG_SIZE, IMG_SIZE, 1), padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_cnn():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    model = build_cnn()
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(MODELS_DIR / "cnn_best.h5"),
                        monitor="val_accuracy",
                        save_best_only=True, verbose=1)
    ]

    print("\nTraining CNN...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # ── Evaluate ───────────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    y_proba = model.predict(X_test, verbose=0).flatten()
    y_pred  = (y_proba > 0.5).astype(int)

    acc = (y_pred == y_test).mean()
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n  Test Accuracy : {acc:.4f}")
    print(f"  ROC-AUC       : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['OK','Defective'])}")

    # ── Plot 1: Training history ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],    label="Train")
    axes[0].plot(history.history["val_accuracy"],label="Val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"],    label="Train")
    axes[1].plot(history.history["val_loss"],label="Val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cnn_training_history.png", dpi=150)
    plt.close()
    print("Saved: results/plots/cnn_training_history.png")

    # ── Plot 2: Confusion matrix ───────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=["OK","Defective"],
                yticklabels=["OK","Defective"])
    plt.title(f"CNN Confusion Matrix — Acc={acc:.3f} AUC={auc:.3f}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cnn_confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: results/plots/cnn_confusion_matrix.png")

    # Save final model
    model.save(str(MODELS_DIR / "cnn_final.h5"))
    print("Saved: models/cnn_final.h5")

    return model, history


if __name__ == "__main__":
    train_cnn()