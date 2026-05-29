import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

DATA_PROC = Path("data/processed")

# ── extract features from one image ───────────────────────────────────
def extract_features(gray_img):
    """
    Extract handcrafted features from a single grayscale image.
    Returns a 1D feature vector.
    """
    img = (gray_img * 255).astype(np.uint8)

    # 1. Statistical features
    mean       = np.mean(img)
    std        = np.std(img)
    skewness   = float(np.mean(((img - mean) / (std + 1e-6)) ** 3))
    kurtosis   = float(np.mean(((img - mean) / (std + 1e-6)) ** 4))

    # 2. Histogram features (16 bins)
    hist, _    = np.histogram(img, bins=16, range=(0, 256))
    hist       = hist.astype(np.float32) / (hist.sum() + 1e-6)

    # 3. Texture — Laplacian variance (sharpness / defect roughness)
    laplacian  = cv2.Laplacian(img, cv2.CV_64F)
    lap_var    = laplacian.var()
    lap_mean   = abs(laplacian.mean())

    # 4. Threshold + contour features
    blurred    = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh  = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _= cv2.findContours(thresh,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    num_contours   = len(contours)
    total_area     = sum(cv2.contourArea(c) for c in contours)
    max_area       = max((cv2.contourArea(c) for c in contours), default=0)

    # 5. Edge density (Canny)
    edges          = cv2.Canny(img, 50, 150)
    edge_density   = edges.sum() / (img.shape[0] * img.shape[1] * 255 + 1e-6)

    # ── assemble feature vector ────────────────────────────────────────
    scalar_features = np.array([
        mean, std, skewness, kurtosis,
        lap_var, lap_mean,
        num_contours, total_area, max_area,
        edge_density
    ], dtype=np.float32)

    feature_vector = np.concatenate([scalar_features, hist])
    return feature_vector  # shape: (26,)


# ── extract features for full dataset ─────────────────────────────────
def build_features():
    print("Loading preprocessed images...")
    X_train = np.load(DATA_PROC / "X_train.npy")
    y_train = np.load(DATA_PROC / "y_train.npy")
    X_test  = np.load(DATA_PROC / "X_test.npy")
    y_test  = np.load(DATA_PROC / "y_test.npy")

    print("Extracting features from training set...")
    X_train_feat = np.array([
        extract_features(img) for img in tqdm(X_train)
    ])

    print("Extracting features from test set...")
    X_test_feat  = np.array([
        extract_features(img) for img in tqdm(X_test)
    ])

    # Save
    np.save(DATA_PROC / "X_train_features.npy", X_train_feat)
    np.save(DATA_PROC / "X_test_features.npy",  X_test_feat)
    np.save(DATA_PROC / "y_train.npy",           y_train)
    np.save(DATA_PROC / "y_test.npy",            y_test)

    print(f"\nFeatures extracted successfully:")
    print(f"  X_train_features: {X_train_feat.shape}")
    print(f"  X_test_features:  {X_test_feat.shape}")
    print(f"  Feature names: mean, std, skewness, kurtosis,")
    print(f"                 laplacian_var, laplacian_mean,")
    print(f"                 num_contours, total_area, max_area,")
    print(f"                 edge_density, hist_bin_0..15")

    return X_train_feat, X_test_feat, y_train, y_test


if __name__ == "__main__":
    build_features()