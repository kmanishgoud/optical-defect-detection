import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DATA_PROC   = Path("data/processed")
MODELS_DIR  = Path("models")
HEATMAPS    = Path("results/heatmaps")
HEATMAPS.mkdir(exist_ok=True)

def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    """
    Compute Grad-CAM heatmap for a single image.
    img_array shape: (1, H, W, 1)
    """
    grad_model = tf.keras.models.Model(
        inputs  = model.input,
        outputs = [model.get_layer(last_conv_layer_name).output,
                   model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads       = tape.gradient(loss, conv_outputs)
    pooled      = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out    = conv_outputs[0]
    heatmap     = conv_out @ pooled[..., tf.newaxis]
    heatmap     = tf.squeeze(heatmap)
    heatmap     = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(heatmap, original_img, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image.
    original_img: grayscale float32 (H, W)
    """
    # Convert grayscale to BGR for colormap overlay
    img_uint8 = (original_img * 255).astype(np.uint8)
    img_bgr   = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_uint8   = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Overlay
    overlaid = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return overlaid


def generate_heatmaps(n_samples=8):
    print("Loading model and data...")
    model = tf.keras.models.load_model(str(MODELS_DIR / "cnn_best.h5"))

    X_test = np.load(DATA_PROC / "X_test.npy")
    y_test = np.load(DATA_PROC / "y_test.npy")

    # Find last conv layer name
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    print(f"  Using last conv layer: {last_conv}")

    # Pick samples — half defective, half ok
    def_idx = np.where(y_test == 1)[0][:n_samples//2]
    ok_idx  = np.where(y_test == 0)[0][:n_samples//2]
    indices = np.concatenate([def_idx, ok_idx])

    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 2.5, 8))
    fig.suptitle("Grad-CAM: Where the CNN Looks to Detect Defects",
                 fontsize=14, fontweight="bold")

    row_labels = ["Original", "Heatmap", "Overlay"]
    for ax, label in zip(axes[:, 0], row_labels):
        ax.set_ylabel(label, fontsize=11, fontweight="bold")

    for col, idx in enumerate(indices):
        img     = X_test[idx]           # (128, 128)
        label   = y_test[idx]
        title   = "DEFECTIVE" if label == 1 else "OK"
        color   = "red" if label == 1 else "green"

        # Prepare for model
        img_input = img[np.newaxis, ..., np.newaxis].astype(np.float32)

        # Grad-CAM
        heatmap  = get_gradcam_heatmap(model, img_input, last_conv)
        overlaid = overlay_heatmap(heatmap, img)
        overlaid_rgb = cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)

        # Heatmap resized
        heatmap_resized = cv2.resize(heatmap,
                                      (img.shape[1], img.shape[0]))

        # Row 0: original
        axes[0, col].imshow(img, cmap="gray")
        axes[0, col].set_title(title, color=color, fontsize=9,
                                fontweight="bold")
        axes[0, col].axis("off")

        # Row 1: heatmap
        axes[1, col].imshow(heatmap_resized, cmap="jet")
        axes[1, col].axis("off")

        # Row 2: overlay
        axes[2, col].imshow(overlaid_rgb)
        axes[2, col].axis("off")

    plt.tight_layout()
    out_path = HEATMAPS / "gradcam_grid.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    print("Grad-CAM heatmaps generated successfully.")


if __name__ == "__main__":
    generate_heatmaps(n_samples=8)
