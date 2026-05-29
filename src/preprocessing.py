import cv2
import numpy as np
import os
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────
DATA_RAW   = Path("data/raw/casting_data/casting_data")
DATA_PROC  = Path("data/processed")
IMG_SIZE   = (128, 128)

TRAIN_DEF  = DATA_RAW / "train" / "def_front"
TRAIN_OK   = DATA_RAW / "train" / "ok_front"
TEST_DEF   = DATA_RAW / "test"  / "def_front"
TEST_OK    = DATA_RAW / "test"  / "ok_front"

# ── single image preprocessor ──────────────────────────────────────────
def preprocess_image(img_path):
    """
    Full preprocessing pipeline for one image.
    Returns: (resized_gray, binary_thresh, contours, original_bgr)
    """
    # 1. Load
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None

    # 2. Resize
    img_bgr = cv2.resize(img_bgr, IMG_SIZE)

    # 3. Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 4. Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 5. Otsu thresholding — automatic, no manual threshold needed
    _, thresh = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6. Contour detection
    contours, _ = cv2.findContours(thresh,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    return gray, thresh, contours, img_bgr


# ── batch loader ───────────────────────────────────────────────────────
def load_dataset_images(folder_path, label, max_images=None):
    """
    Load and preprocess all images in a folder.
    Returns list of (gray_array, label) tuples.
    """
    folder = Path(folder_path)
    images, labels = [], []

    files = list(folder.glob("*.jpeg")) + \
            list(folder.glob("*.jpg"))  + \
            list(folder.glob("*.png"))

    if max_images:
        files = files[:max_images]

    for f in files:
        result = preprocess_image(f)
        if result is not None:
            gray, _, _, _ = result
            images.append(gray)
            labels.append(label)

    print(f"Loaded {len(images)} images from {folder.name} (label={label})")
    return images, labels


# ── build full arrays ──────────────────────────────────────────────────
def build_dataset():
    """
    Build train and test numpy arrays from raw data.
    label: 0 = ok,  1 = defective
    """
    print("Building dataset...")

    tr_def_imgs, tr_def_lbs = load_dataset_images(TRAIN_DEF, label=1)
    tr_ok_imgs,  tr_ok_lbs  = load_dataset_images(TRAIN_OK,  label=0)
    te_def_imgs, te_def_lbs = load_dataset_images(TEST_DEF,  label=1)
    te_ok_imgs,  te_ok_lbs  = load_dataset_images(TEST_OK,   label=0)

    X_train = np.array(tr_def_imgs + tr_ok_imgs, dtype=np.float32) / 255.0
    y_train = np.array(tr_def_lbs  + tr_ok_lbs,  dtype=np.int32)

    X_test  = np.array(te_def_imgs + te_ok_imgs,  dtype=np.float32) / 255.0
    y_test  = np.array(te_def_lbs  + te_ok_lbs,   dtype=np.int32)

    # Save
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    np.save(DATA_PROC / "X_train.npy", X_train)
    np.save(DATA_PROC / "y_train.npy", y_train)
    np.save(DATA_PROC / "X_test.npy",  X_test)
    np.save(DATA_PROC / "y_test.npy",  y_test)

    print(f"\nDataset built successfully:")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")
    print(f"  Saved to {DATA_PROC}")

    return X_train, y_train, X_test, y_test


# ── run directly ───────────────────────────────────────────────────────
if __name__ == "__main__":
    build_dataset()
