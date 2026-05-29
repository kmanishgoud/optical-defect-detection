import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import joblib
import warnings
warnings.filterwarnings("ignore")

DATA_PROC  = Path("data/processed")
MODELS_DIR = Path("models")
PLOTS_DIR  = Path("results/plots")
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

def load_features():
    X_train = np.load(DATA_PROC / "X_train_features.npy")
    X_test  = np.load(DATA_PROC / "X_test_features.npy")
    y_train = np.load(DATA_PROC / "y_train.npy")
    y_test  = np.load(DATA_PROC / "y_test.npy")
    return X_train, X_test, y_train, y_test

def train_models():
    print("Loading features...")
    X_train, X_test, y_train, y_test = load_features()

    # Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    # Define models
    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=100,
                                                       random_state=42,
                                                       n_jobs=-1),
        "SVM":                 SVC(kernel="rbf", probability=True,
                                   random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100,
                                                           random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_sc, y_train)

        y_pred  = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]

        acc     = accuracy_score(y_test, y_pred)
        auc     = roc_auc_score(y_test, y_proba)
        results[name] = {"model": model, "acc": acc, "auc": auc,
                         "y_pred": y_pred, "y_proba": y_proba}

        print(f"  Accuracy : {acc:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['OK','Defective'])}")

        # Save model
        joblib.dump(model, MODELS_DIR / f"{name.replace(' ','_')}.pkl")

    # ── Plot 1: Confusion matrices ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["OK","Defective"],
                    yticklabels=["OK","Defective"])
        ax.set_title(f"{name}\nAcc={res['acc']:.3f}")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close()
    print("\nSaved: results/plots/confusion_matrices.png")

    # ── Plot 2: ROC curves ─────────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
    plt.plot([0,1],[0,1],"k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — ML Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curves.png", dpi=150)
    plt.close()
    print("Saved: results/plots/roc_curves.png")

    # ── Plot 3: Feature importance (Random Forest) ─────────────────────
    feature_names = (["mean","std","skewness","kurtosis",
                       "lap_var","lap_mean",
                       "num_contours","total_area","max_area",
                       "edge_density"] +
                     [f"hist_{i}" for i in range(16)])

    rf = results["Random Forest"]["model"]
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(importances)),
            importances[indices], color="steelblue")
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in indices],
               rotation=45, ha="right")
    plt.title("Feature Importances — Random Forest")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importances.png", dpi=150)
    plt.close()
    print("Saved: results/plots/feature_importances.png")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "="*45)
    print("SUMMARY")
    print("="*45)
    best = max(results, key=lambda k: results[k]["auc"])
    for name, res in results.items():
        marker = " ← best" if name == best else ""
        print(f"  {name:25s}  Acc={res['acc']:.4f}  AUC={res['auc']:.4f}{marker}")

if __name__ == "__main__":
    train_models()
