import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
from src.features import extract_features
from src.gradcam import get_gradcam_heatmap, overlay_heatmap

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Optical Defect Detection",
    page_icon="🔬",
    layout="wide"
)

# ── Load models (cached) ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    cnn     = tf.keras.models.load_model("models/cnn_best.h5")
    rf      = joblib.load("models/Random_Forest.pkl")
    svm     = joblib.load("models/SVM.pkl")
    gb      = joblib.load("models/Gradient_Boosting.pkl")
    scaler  = joblib.load("models/scaler.pkl")
    return cnn, rf, svm, gb, scaler

# ── Preprocess uploaded image ──────────────────────────────────────────
def preprocess_uploaded(image_bytes):
    img_array = np.frombuffer(image_bytes, np.uint8)
    img_bgr   = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_bgr   = cv2.resize(img_bgr, (128, 128))
    gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_f    = gray.astype(np.float32) / 255.0
    return gray_f, img_bgr

# ── Main app ───────────────────────────────────────────────────────────
def main():
    # Header
    st.title("🔬 Automated Optical Defect Detection System")
    st.markdown("""
    **AI-powered visual inspection** using Computer Vision & Machine Learning.  
    Upload an image of an industrial component to get an instant defect analysis.
    """)

    st.divider()

    # Load models
    with st.spinner("Loading models..."):
        cnn, rf, svm, gb, scaler = load_models()

    # Sidebar
    st.sidebar.title("⚙️ Settings")
    model_choice = st.sidebar.selectbox(
        "Select ML Model",
        ["CNN (Deep Learning)", "SVM", "Random Forest", "Gradient Boosting"]
    )
    show_gradcam = st.sidebar.checkbox("Show Grad-CAM Heatmap", value=True)
    show_features = st.sidebar.checkbox("Show Feature Analysis", value=True)

    st.sidebar.divider()
    st.sidebar.markdown("### 📊 Model Performance")
    st.sidebar.markdown("""
    | Model | Accuracy | AUC |
    |---|---|---|
    | CNN | 93.8% | 0.984 |
    | SVM | 100% | 1.000 |
    | Random Forest | 99.3% | 0.999 |
    | Gradient Boosting | 98.7% | 0.999 |
    """)

    # Upload
    uploaded = st.file_uploader(
        "Upload a component image (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:
        image_bytes = uploaded.read()
        gray_f, img_bgr = preprocess_uploaded(image_bytes)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                     caption="Input Component", use_column_width=True)

        # ── Prediction ─────────────────────────────────────────────────
        with col2:
            st.subheader("🤖 Prediction")

            if model_choice == "CNN (Deep Learning)":
                img_input = gray_f[np.newaxis, ..., np.newaxis]
                proba     = float(cnn.predict(img_input, verbose=0)[0][0])
                model_used = cnn
            else:
                features  = extract_features(gray_f).reshape(1, -1)
                features_sc = scaler.transform(features)
                if model_choice == "SVM":
                    proba = float(svm.predict_proba(features_sc)[0][1])
                elif model_choice == "Random Forest":
                    proba = float(rf.predict_proba(features_sc)[0][1])
                else:
                    proba = float(gb.predict_proba(features_sc)[0][1])
                model_used = None

            label     = "DEFECTIVE" if proba > 0.5 else "OK"
            confidence = proba if proba > 0.5 else 1 - proba

            if label == "DEFECTIVE":
                st.error(f"### ❌ {label}")
            else:
                st.success(f"### ✅ {label}")

            st.metric("Defect Probability", f"{proba:.1%}")
            st.metric("Confidence",         f"{confidence:.1%}")
            st.progress(float(proba))

            st.markdown(f"**Model used:** {model_choice}")

        st.divider()

        # ── Grad-CAM ───────────────────────────────────────────────────
        if show_gradcam and model_choice == "CNN (Deep Learning)":
            st.subheader("🔥 Grad-CAM Attention Heatmap")
            st.markdown("""
            *Visualising where the CNN focuses its attention —
            red/yellow regions are most influential for the decision.*
            """)

            img_input  = gray_f[np.newaxis, ..., np.newaxis]
            last_conv  = None
            for layer in reversed(cnn.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv = layer.name
                    break

            heatmap    = get_gradcam_heatmap(cnn, img_input, last_conv)
            overlaid   = overlay_heatmap(heatmap, gray_f)
            import cv2 as cv
            overlaid_rgb = cv.cvtColor(overlaid, cv.COLOR_BGR2RGB)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(gray_f, caption="Original (grayscale)",
                         clamp=True, use_column_width=True)
            with c2:
                fig, ax = plt.subplots()
                hm_resized = cv.resize(heatmap, (128, 128))
                ax.imshow(hm_resized, cmap="jet")
                ax.axis("off")
                st.pyplot(fig)
                st.caption("Attention Heatmap")
                plt.close()
            with c3:
                st.image(overlaid_rgb, caption="Overlay",
                         use_column_width=True)

        # ── Feature Analysis ───────────────────────────────────────────
        if show_features:
            st.subheader("📊 Feature Analysis")
            features = extract_features(gray_f)
            feat_names = (["mean","std","skewness","kurtosis",
                           "lap_var","lap_mean",
                           "num_contours","total_area","max_area",
                           "edge_density"] +
                          [f"hist_{i}" for i in range(16)])

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Scalar Features**")
                scalar_data = {feat_names[i]: float(features[i])
                               for i in range(10)}
                st.dataframe(scalar_data, use_container_width=True)

            with col_b:
                st.markdown("**Histogram Distribution**")
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(range(16), features[10:], color="steelblue")
                ax.set_xlabel("Bin")
                ax.set_ylabel("Frequency")
                ax.set_title("Pixel Intensity Histogram")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    else:
        # Show sample results when no image uploaded
        st.info("👆 Upload an image to get started")

        st.subheader("📈 Model Performance Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CNN Accuracy",      "93.8%", "Deep Learning")
        c2.metric("SVM Accuracy",      "100%",  "Best Classical")
        c3.metric("Random Forest",     "99.3%", "Ensemble")
        c4.metric("Gradient Boosting", "98.7%", "Ensemble")

        st.subheader("🖼️ Sample Grad-CAM Results")
        heatmap_path = Path("results/heatmaps/gradcam_grid.png")
        if heatmap_path.exists():
            st.image(str(heatmap_path),
                     caption="Grad-CAM attention maps on test images",
                     use_column_width=True)

if __name__ == "__main__":
    main()