import os
import warnings
import numpy as np
import pandas as pd
import librosa
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# General configuration
# ============================================================

warnings.filterwarnings("ignore", category=UserWarning)

AUDIO_DIR = "data/raw"
LABELS_PATH = "data/metadata/labels_clean.csv"

RESULTS_DIR = "experiments/exp_08_mfcc_svm_augmentation_strict"
RESULTS_PATH = os.path.join(RESULTS_DIR, "results.txt")

MODEL_OUT = "models/mfcc_svm_augmentation_strict/mfcc_svm_augmentation_strict.joblib"

SAMPLE_RATE = 16000
N_MFCC = 20
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


# ============================================================
# Audio loading
# ============================================================

def load_audio(audio_path: str, sr: int = 16000) -> np.ndarray:
    """
    Load one audio file and resample it to 16 kHz.
    """

    audio, _ = librosa.load(audio_path, sr=sr)
    return audio.astype(np.float32)


# ============================================================
# Data augmentation
# ============================================================

def add_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """
    Add small Gaussian noise to the signal.
    """

    noise = np.random.randn(len(audio)).astype(np.float32)
    return (audio + noise_factor * noise).astype(np.float32)


def change_amplitude(audio: np.ndarray, factor: float = 1.2) -> np.ndarray:
    """
    Change the global amplitude of the signal.
    """

    return (audio * factor).astype(np.float32)


def time_stretch_safe(audio: np.ndarray, rate: float = 1.05) -> np.ndarray:
    """
    Apply time stretching.

    If librosa fails for a very short signal, the original audio is returned.
    """

    try:
        return librosa.effects.time_stretch(audio, rate=rate).astype(np.float32)
    except Exception:
        return audio.astype(np.float32)


def pitch_shift_safe(audio: np.ndarray, sr: int = 16000, n_steps: int = 1) -> np.ndarray:
    """
    Apply pitch shifting.

    If librosa fails for a very short signal, the original audio is returned.
    """

    try:
        return librosa.effects.pitch_shift(
            audio,
            sr=sr,
            n_steps=n_steps
        ).astype(np.float32)
    except Exception:
        return audio.astype(np.float32)


def augment_audio(audio: np.ndarray, sr: int = 16000):
    """
    Create augmented versions of one audio signal.

    Important:
    These augmentations must be applied only to the training fold.
    The test fold must always contain original audios only.
    """

    return [
        ("original", audio.astype(np.float32)),
        ("noise_0005", add_noise(audio, noise_factor=0.005)),
        ("noise_0010", add_noise(audio, noise_factor=0.010)),
        ("amp_080", change_amplitude(audio, factor=0.8)),
        ("amp_120", change_amplitude(audio, factor=1.2)),
        ("stretch_095", time_stretch_safe(audio, rate=0.95)),
        ("stretch_105", time_stretch_safe(audio, rate=1.05)),
        ("pitch_minus1", pitch_shift_safe(audio, sr=sr, n_steps=-1)),
        ("pitch_plus1", pitch_shift_safe(audio, sr=sr, n_steps=1)),
    ]


# ============================================================
# Feature extraction
# ============================================================

def extract_mfcc_delta_features(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extract MFCC + delta + delta-delta features.

    For each feature type, we compute:
    - mean over time
    - standard deviation over time

    With n_mfcc = 20:
    MFCC mean/std        = 40
    Delta mean/std       = 40
    Delta-delta mean/std = 40
    Total                = 120
    """

    audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)

    if len(audio_trimmed) > int(0.2 * sr):
        audio = audio_trimmed

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    delta = librosa.feature.delta(mfcc)
    delta_delta = librosa.feature.delta(mfcc, order=2)

    def summarize(matrix: np.ndarray) -> np.ndarray:
        mean = np.mean(matrix, axis=1)
        std = np.std(matrix, axis=1)
        return np.concatenate([mean, std])

    return np.concatenate([
        summarize(mfcc),
        summarize(delta),
        summarize(delta_delta)
    ])


# ============================================================
# Dataset preparation
# ============================================================

def load_original_dataset() -> pd.DataFrame:
    """
    Load labels_clean.csv and keep only rows with existing audio files.
    """

    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

    df = pd.read_csv(LABELS_PATH)

    required_columns = {"filename", "label_id"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"labels_clean.csv must contain columns {required_columns}. "
            f"Missing columns: {missing_columns}"
        )

    rows = []
    missing_files = []

    for _, row in df.iterrows():
        filename = str(row["filename"]).strip()
        label = str(row["label_id"]).strip()

        audio_path = os.path.join(AUDIO_DIR, filename)

        if not os.path.exists(audio_path):
            missing_files.append(audio_path)
            continue

        rows.append({
            "filename": filename,
            "label_id": label,
            "audio_path": audio_path
        })

    if missing_files:
        print("\nMissing audio files:")
        for path in missing_files:
            print("-", path)

    return pd.DataFrame(rows)


def filter_classes_with_enough_samples(df: pd.DataFrame, min_samples: int = 2):
    """
    Remove classes with fewer than min_samples examples.

    StratifiedKFold with n_splits=2 requires every evaluated class
    to have at least 2 examples.
    """

    class_counts = df["label_id"].value_counts().sort_index()

    valid_classes = class_counts[class_counts >= min_samples].index
    removed_classes = class_counts[class_counts < min_samples]

    filtered_df = df[df["label_id"].isin(valid_classes)].copy()
    filtered_df = filtered_df.reset_index(drop=True)

    return filtered_df, class_counts, removed_classes


# ============================================================
# Fold feature building
# ============================================================

def build_train_features(train_df: pd.DataFrame):
    """
    Build training features.

    For each original training audio:
    - keep the original signal
    - add augmented versions

    This function is called inside each fold to avoid data leakage.
    """

    X_train = []
    y_train = []
    source_files = []
    augmentation_names = []

    for _, row in train_df.iterrows():
        audio_path = row["audio_path"]
        label = row["label_id"]
        filename = row["filename"]

        audio = load_audio(audio_path, sr=SAMPLE_RATE)
        augmented_versions = augment_audio(audio, sr=SAMPLE_RATE)

        for aug_name, aug_audio in augmented_versions:
            features = extract_mfcc_delta_features(
                aug_audio,
                sr=SAMPLE_RATE
            )

            X_train.append(features)
            y_train.append(label)
            source_files.append(filename)
            augmentation_names.append(aug_name)

    return (
        np.array(X_train),
        np.array(y_train),
        np.array(source_files),
        np.array(augmentation_names)
    )


def build_test_features(test_df: pd.DataFrame):
    """
    Build test features.

    Important:
    The test fold uses original audios only.
    No augmentation is applied to test data.
    """

    X_test = []
    y_test = []
    source_files = []

    for _, row in test_df.iterrows():
        audio_path = row["audio_path"]
        label = row["label_id"]
        filename = row["filename"]

        audio = load_audio(audio_path, sr=SAMPLE_RATE)
        features = extract_mfcc_delta_features(
            audio,
            sr=SAMPLE_RATE
        )

        X_test.append(features)
        y_test.append(label)
        source_files.append(filename)

    return np.array(X_test), np.array(y_test), np.array(source_files)


# ============================================================
# Model evaluation
# ============================================================

def get_svm_configs():
    """
    SVM configurations tested in the experiment.
    """

    return [
        {"kernel": "rbf", "C": 0.1, "gamma": "scale"},
        {"kernel": "rbf", "C": 1, "gamma": "scale"},
        {"kernel": "rbf", "C": 10, "gamma": "scale"},
        {"kernel": "rbf", "C": 100, "gamma": "scale"},
        {"kernel": "linear", "C": 1, "gamma": "scale"},
        {"kernel": "linear", "C": 10, "gamma": "scale"},
    ]


def make_svm_pipeline(config: dict):
    """
    Create a scaler + SVM pipeline.
    """

    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=config["kernel"],
            C=config["C"],
            gamma=config["gamma"],
            class_weight="balanced"
        ))
    ])


def run_strict_augmentation_cv(df_eval: pd.DataFrame, label_encoder: LabelEncoder):
    """
    Run strict train-only augmentation cross-validation.

    Split is done on original audios only.
    Augmentation is applied only after the split, and only to the train fold.
    """

    y_encoded_all = label_encoder.transform(df_eval["label_id"].values)

    class_counts = df_eval["label_id"].value_counts()
    min_class_count = class_counts.min()

    n_splits = min(2, min_class_count)

    if n_splits < 2:
        raise ValueError(
            "StratifiedKFold is impossible: at least one evaluated class "
            "has fewer than 2 samples."
        )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    configs = get_svm_configs()

    all_results = []
    best_result = None
    best_model = None
    best_fold_macro_f1 = -1.0

    for config in configs:
        accuracies = []
        macro_f1s = []
        fold_reports = []
        fold_confusions = []
        fold_train_sizes = []
        fold_test_sizes = []

        print("\n========================================")
        print("Testing configuration:", config)
        print("========================================")

        for fold_id, (train_idx, test_idx) in enumerate(skf.split(df_eval, y_encoded_all), start=1):
            print(f"\nFold {fold_id}/{n_splits}")

            train_df = df_eval.iloc[train_idx].reset_index(drop=True)
            test_df = df_eval.iloc[test_idx].reset_index(drop=True)

            X_train, y_train_labels, train_sources, train_augs = build_train_features(train_df)
            X_test, y_test_labels, test_sources = build_test_features(test_df)

            y_train = label_encoder.transform(y_train_labels)
            y_test = label_encoder.transform(y_test_labels)

            print("Train original audios:", len(train_df))
            print("Train samples after augmentation:", len(X_train))
            print("Test original audios:", len(test_df))

            pipeline = make_svm_pipeline(config)
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0
            )

            accuracies.append(accuracy)
            macro_f1s.append(macro_f1)
            fold_train_sizes.append((len(train_df), len(X_train)))
            fold_test_sizes.append(len(test_df))

            labels_present = sorted(np.unique(np.concatenate([y_test, y_pred])))
            target_names = label_encoder.inverse_transform(labels_present)

            report = classification_report(
                y_test,
                y_pred,
                labels=labels_present,
                target_names=target_names,
                zero_division=0
            )

            confusion = confusion_matrix(
                y_test,
                y_pred,
                labels=labels_present
            )

            fold_reports.append((fold_id, report))
            fold_confusions.append((fold_id, confusion, target_names))

            print(f"Accuracy fold {fold_id}: {accuracy:.4f}")
            print(f"Macro F1 fold {fold_id}: {macro_f1:.4f}")

            # Keep the best single-fold model only for saving a usable model.
            # Reported scores remain the cross-validation averages.
            if macro_f1 > best_fold_macro_f1:
                best_fold_macro_f1 = macro_f1
                best_model = pipeline

        result = {
            "config": config,
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "macro_f1_mean": float(np.mean(macro_f1s)),
            "macro_f1_std": float(np.std(macro_f1s)),
            "fold_reports": fold_reports,
            "fold_confusions": fold_confusions,
            "fold_train_sizes": fold_train_sizes,
            "fold_test_sizes": fold_test_sizes,
            "best_fold_macro_f1": float(max(macro_f1s)),
        }

        all_results.append(result)

        print("\nConfiguration summary:")
        print(f"Accuracy mean : {result['accuracy_mean']:.4f}")
        print(f"Accuracy std  : {result['accuracy_std']:.4f}")
        print(f"Macro F1 mean : {result['macro_f1_mean']:.4f}")
        print(f"Macro F1 std  : {result['macro_f1_std']:.4f}")

        if best_result is None or result["macro_f1_mean"] > best_result["macro_f1_mean"]:
            best_result = result

    return all_results, best_result, best_model, n_splits


# ============================================================
# Final model training
# ============================================================

def train_final_model_with_augmentation(df_eval: pd.DataFrame, label_encoder: LabelEncoder, best_config: dict):
    """
    Train a final model on all evaluable audios with train-style augmentation.

    This model is saved for later inference, but the reported scores are only
    the cross-validation scores above.
    """

    X_train, y_train_labels, train_sources, train_augs = build_train_features(df_eval)
    y_train = label_encoder.transform(y_train_labels)

    pipeline = make_svm_pipeline(best_config)
    pipeline.fit(X_train, y_train)

    return pipeline, len(X_train)


# ============================================================
# Results saving
# ============================================================

def save_results(
    all_results,
    best_result,
    raw_class_counts,
    evaluated_class_counts,
    removed_classes,
    n_splits,
    final_train_samples
):
    """
    Save a clear and transparent results file.
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("Experiment: Strict Train-Only Augmentation + MFCC Dynamic Features + SVM\n")
        f.write("========================================================================\n\n")

        f.write("Description:\n")
        f.write(
            "This experiment evaluates data augmentation under a strict protocol. "
            "The split is performed on original audio files only. Augmentation is "
            "then applied exclusively to the training fold. The test fold always "
            "contains original, non-augmented audio files. This avoids data leakage.\n\n"
        )

        f.write("Features:\n")
        f.write(f"- Sample rate: {SAMPLE_RATE} Hz\n")
        f.write(f"- Number of MFCC coefficients: {N_MFCC}\n")
        f.write("- Feature type: MFCC + delta + delta-delta\n")
        f.write("- Statistics: mean + standard deviation over time\n")
        f.write("- Feature dimension: 120\n\n")

        f.write("Augmentations used on training folds only:\n")
        for aug_name in [
            "original",
            "noise_0005",
            "noise_0010",
            "amp_080",
            "amp_120",
            "stretch_095",
            "stretch_105",
            "pitch_minus1",
            "pitch_plus1",
        ]:
            f.write(f"- {aug_name}\n")
        f.write("\n")

        f.write("Cross-validation:\n")
        f.write(f"- StratifiedKFold with n_splits = {n_splits}\n")
        f.write("- Classes with fewer than 2 samples are excluded from evaluation\n")
        f.write("- Test data are never augmented\n\n")

        f.write("Original class distribution:\n")
        f.write(raw_class_counts.to_string())
        f.write("\n\n")

        f.write("Removed classes with fewer than 2 samples:\n")
        if len(removed_classes) > 0:
            f.write(removed_classes.to_string())
        else:
            f.write("None")
        f.write("\n\n")

        f.write("Evaluated class distribution:\n")
        f.write(evaluated_class_counts.to_string())
        f.write("\n\n")

        f.write("All tested SVM configurations:\n")
        for result in all_results:
            f.write("\n")
            f.write(f"Config: {result['config']}\n")
            f.write(f"Accuracy mean: {result['accuracy_mean']:.4f}\n")
            f.write(f"Accuracy std : {result['accuracy_std']:.4f}\n")
            f.write(f"Macro F1 mean: {result['macro_f1_mean']:.4f}\n")
            f.write(f"Macro F1 std : {result['macro_f1_std']:.4f}\n")

        f.write("\n\nBest configuration:\n")
        f.write(str(best_result["config"]))
        f.write("\n")
        f.write(f"Best Accuracy mean: {best_result['accuracy_mean']:.4f}\n")
        f.write(f"Best Accuracy std : {best_result['accuracy_std']:.4f}\n")
        f.write(f"Best Macro F1 mean: {best_result['macro_f1_mean']:.4f}\n")
        f.write(f"Best Macro F1 std : {best_result['macro_f1_std']:.4f}\n")

        f.write("\n\nFinal saved model:\n")
        f.write(
            "The final saved model is trained on all evaluable original audios "
            "with train-style augmentation.\n"
        )
        f.write(f"Final augmented training samples: {final_train_samples}\n")

        f.write("\n\nDetailed classification reports for the best configuration:\n")
        for fold_id, report in best_result["fold_reports"]:
            f.write("\n")
            f.write(f"Fold {fold_id}\n")
            f.write("----------------------------------------\n")
            f.write(report)
            f.write("\n")

        f.write("\n\nConfusion matrices for the best configuration:\n")
        for fold_id, confusion, names in best_result["fold_confusions"]:
            f.write("\n")
            f.write(f"Fold {fold_id}\n")
            f.write("----------------------------------------\n")
            f.write("Labels: " + ", ".join(names) + "\n")
            f.write(str(confusion))
            f.write("\n")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    df = load_original_dataset()

    print("\nRaw dataset:")
    print("Number of original audios:", len(df))
    print("Number of classes:", df["label_id"].nunique())

    raw_class_counts = df["label_id"].value_counts().sort_index()

    print("\nRaw class distribution:")
    print(raw_class_counts)

    df_eval, raw_counts, removed_classes = filter_classes_with_enough_samples(
        df,
        min_samples=2
    )

    evaluated_class_counts = df_eval["label_id"].value_counts().sort_index()

    print("\nDataset used for evaluation:")
    print("Number of original audios:", len(df_eval))
    print("Number of classes:", df_eval["label_id"].nunique())

    print("\nEvaluated class distribution:")
    print(evaluated_class_counts)

    print("\nRemoved classes with fewer than 2 samples:")
    print(removed_classes if len(removed_classes) > 0 else "None")

    label_encoder = LabelEncoder()
    label_encoder.fit(df_eval["label_id"].values)

    all_results, best_result, _, n_splits = run_strict_augmentation_cv(
        df_eval=df_eval,
        label_encoder=label_encoder
    )

    print("\n========================================")
    print("Best configuration")
    print("========================================")
    print(best_result["config"])
    print(f"Accuracy mean : {best_result['accuracy_mean']:.4f}")
    print(f"Accuracy std  : {best_result['accuracy_std']:.4f}")
    print(f"Macro F1 mean : {best_result['macro_f1_mean']:.4f}")
    print(f"Macro F1 std  : {best_result['macro_f1_std']:.4f}")

    final_model, final_train_samples = train_final_model_with_augmentation(
        df_eval=df_eval,
        label_encoder=label_encoder,
        best_config=best_result["config"]
    )

    joblib.dump({
        "pipeline": final_model,
        "label_encoder": label_encoder,
        "feature_type": "mfcc_delta_deltadelta_mean_std",
        "augmentation_protocol": "strict_train_only",
        "n_mfcc": N_MFCC,
        "sample_rate": SAMPLE_RATE,
        "removed_classes": removed_classes.to_dict(),
        "best_config": best_result["config"],
        "evaluated_labels": list(label_encoder.classes_),
    }, MODEL_OUT)

    save_results(
        all_results=all_results,
        best_result=best_result,
        raw_class_counts=raw_counts,
        evaluated_class_counts=evaluated_class_counts,
        removed_classes=removed_classes,
        n_splits=n_splits,
        final_train_samples=final_train_samples
    )

    print(f"\nResults saved: {RESULTS_PATH}")
    print(f"Final model saved: {MODEL_OUT}")


if __name__ == "__main__":
    main()