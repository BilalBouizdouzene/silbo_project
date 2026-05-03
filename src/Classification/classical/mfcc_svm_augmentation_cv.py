import os
import warnings
import numpy as np
import pandas as pd
import librosa
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score


warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# Paths
# =========================

AUDIO_DIR = "data/raw"
LABELS_PATH = "data/metadata/labels_clean.csv"

RESULTS_DIR = "experiments/exp_07_mfcc_svm_augmentation"
MODEL_OUT = "models/mfcc_svm_augmentation/mfcc_svm_augmentation.joblib"

# =========================
# Config
# =========================

SAMPLE_RATE = 16000
N_MFCC = 20
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


def load_audio(audio_path, sr=16000):
    y, _ = librosa.load(audio_path, sr=sr)
    return y


def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    augmented = y + noise_factor * noise
    return augmented.astype(np.float32)


def change_amplitude(y, factor=1.2):
    return (y * factor).astype(np.float32)


def time_stretch_safe(y, rate=1.05):
    try:
        return librosa.effects.time_stretch(y, rate=rate).astype(np.float32)
    except Exception:
        return y.astype(np.float32)


def pitch_shift_safe(y, sr=16000, n_steps=1):
    try:
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps).astype(np.float32)
    except Exception:
        return y.astype(np.float32)


def augment_audio(y, sr=16000):
    """
    Return list of (augmentation_name, audio).
    The original is included.
    """
    augmented_versions = []

    augmented_versions.append(("original", y.astype(np.float32)))

    augmented_versions.append(("noise_0005", add_noise(y, noise_factor=0.005)))
    augmented_versions.append(("noise_0010", add_noise(y, noise_factor=0.010)))

    augmented_versions.append(("amp_080", change_amplitude(y, factor=0.8)))
    augmented_versions.append(("amp_120", change_amplitude(y, factor=1.2)))

    augmented_versions.append(("stretch_095", time_stretch_safe(y, rate=0.95)))
    augmented_versions.append(("stretch_105", time_stretch_safe(y, rate=1.05)))

    augmented_versions.append(("pitch_minus1", pitch_shift_safe(y, sr=sr, n_steps=-1)))
    augmented_versions.append(("pitch_plus1", pitch_shift_safe(y, sr=sr, n_steps=1)))

    return augmented_versions


def extract_mfcc(y, sr=16000, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    return np.concatenate([mfcc_mean, mfcc_std])


def build_augmented_dataset():
    df = pd.read_csv(LABELS_PATH)

    if "filename" not in df.columns or "label_id" not in df.columns:
        raise ValueError("labels_clean.csv must contain filename and label_id columns.")

    X = []
    y = []
    filenames = []
    augmentations = []
    missing = []

    for _, row in df.iterrows():
        filename = str(row["filename"]).strip()
        label = str(row["label_id"]).strip()

        audio_path = os.path.join(AUDIO_DIR, filename)

        if not os.path.exists(audio_path):
            missing.append(audio_path)
            continue

        audio = load_audio(audio_path, sr=SAMPLE_RATE)

        augmented_versions = augment_audio(audio, sr=SAMPLE_RATE)

        for aug_name, aug_audio in augmented_versions:
            features = extract_mfcc(aug_audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

            X.append(features)
            y.append(label)
            filenames.append(filename)
            augmentations.append(aug_name)

    if missing:
        print("\nAudios introuvables :")
        for m in missing:
            print("-", m)

    X = np.array(X)
    y = np.array(y)
    filenames = np.array(filenames)
    augmentations = np.array(augmentations)

    return X, y, filenames, augmentations


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    print("Building augmented dataset...")
    X, y, filenames, augmentations = build_augmented_dataset()

    print("\nNumber of augmented samples:", len(X))
    print("Feature dimension:", X.shape[1])
    print("Number of classes:", len(set(y)))

    class_counts = pd.Series(y).value_counts().sort_index()

    print("\nClass distribution after augmentation:")
    print(class_counts)

    print("\nAugmentation distribution:")
    print(pd.Series(augmentations).value_counts().sort_index())

    min_class_count = class_counts.min()
    n_splits = min(2, min_class_count)

    if n_splits < 2:
        raise ValueError("At least one class has fewer than 2 samples.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale"))
    ])

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    accuracy_scores = cross_val_score(
        pipeline,
        X,
        y_encoded,
        cv=cv,
        scoring="accuracy"
    )

    macro_f1_scores = cross_val_score(
        pipeline,
        X,
        y_encoded,
        cv=cv,
        scoring=make_scorer(f1_score, average="macro", zero_division=0)
    )

    print("\n=== Data Augmentation + MFCC + SVM Cross-Validation Results ===")
    print(f"Accuracy mean: {accuracy_scores.mean():.4f}")
    print(f"Accuracy std : {accuracy_scores.std():.4f}")
    print(f"Macro F1 mean: {macro_f1_scores.mean():.4f}")
    print(f"Macro F1 std : {macro_f1_scores.std():.4f}")

    # Train final model on all augmented data
    pipeline.fit(X, y_encoded)

    joblib.dump({
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "n_mfcc": N_MFCC,
        "sample_rate": SAMPLE_RATE,
        "augmentations": sorted(set(augmentations))
    }, MODEL_OUT)

    results_path = os.path.join(RESULTS_DIR, "results.txt")

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("Experiment: Data Augmentation + MFCC + SVM\n")
        f.write("==========================================\n\n")
        f.write(f"Original audios: {len(set(filenames))}\n")
        f.write(f"Augmented samples: {len(X)}\n")
        f.write(f"Number of classes: {len(set(y))}\n")
        f.write(f"Features: MFCC mean + std, n_mfcc={N_MFCC}\n")
        f.write("Model: SVM RBF, C=10, gamma=scale\n")
        f.write(f"Cross-validation: StratifiedKFold, n_splits={n_splits}\n\n")

        f.write("Augmentations used:\n")
        for aug in sorted(set(augmentations)):
            f.write(f"- {aug}\n")

        f.write("\nClass distribution after augmentation:\n")
        f.write(class_counts.to_string())
        f.write("\n\n")

        f.write("Augmentation distribution:\n")
        f.write(pd.Series(augmentations).value_counts().sort_index().to_string())
        f.write("\n\n")

        f.write("Results:\n")
        f.write(f"Accuracy mean: {accuracy_scores.mean():.4f}\n")
        f.write(f"Accuracy std: {accuracy_scores.std():.4f}\n")
        f.write(f"Macro F1 mean: {macro_f1_scores.mean():.4f}\n")
        f.write(f"Macro F1 std: {macro_f1_scores.std():.4f}\n")

        f.write("\nImportant note:\n")
        f.write(
            "This is a preliminary augmentation experiment. "
            "Augmentation was applied before cross-validation. "
            "For a stricter protocol, augmentation should be applied only "
            "on the training fold to avoid data leakage.\n"
        )

    print(f"\nResults saved: {results_path}")
    print(f"Final model saved: {MODEL_OUT}")


if __name__ == "__main__":
    main()