import os
import warnings
import numpy as np
import pandas as pd
import librosa
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ============================================================
# Configuration
# ============================================================

warnings.filterwarnings("ignore", category=UserWarning)

AUDIO_DIR = "data/raw"
LABELS_PATH = "data/metadata/labels_clean.csv"

RESULTS_DIR = "experiments/exp_01_mfcc_svm"
RESULTS_PATH = os.path.join(RESULTS_DIR, "results.txt")

MODEL_OUT = "models/svm/mfcc_svm_cv.joblib"

SAMPLE_RATE = 16000
N_MFCC = 20
RANDOM_STATE = 42


# ============================================================
# Feature extraction
# ============================================================

def extract_mfcc_delta_features(audio_path: str) -> np.ndarray:
    """
    Extract MFCC + delta + delta-delta features from one audio file.

    For each feature matrix, we compute:
    - mean over time
    - standard deviation over time

    With 20 MFCC coefficients:
    MFCC mean/std           = 40
    Delta mean/std          = 40
    Delta-delta mean/std    = 40
    Total                   = 120
    """

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    # Remove leading and trailing silence if possible
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)

    # Keep trimmed signal only if it is not too short
    if len(y_trimmed) > int(0.2 * SAMPLE_RATE):
        y = y_trimmed

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC
    )

    delta = librosa.feature.delta(mfcc)
    delta_delta = librosa.feature.delta(mfcc, order=2)

    def summarize(matrix: np.ndarray) -> np.ndarray:
        mean = np.mean(matrix, axis=1)
        std = np.std(matrix, axis=1)
        return np.concatenate([mean, std])

    features = np.concatenate([
        summarize(mfcc),
        summarize(delta),
        summarize(delta_delta)
    ])

    return features


# ============================================================
# Dataset loading
# ============================================================

def load_dataset():
    """
    Load labels_clean.csv and extract features for all available audios.
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

    X = []
    y = []
    filenames = []
    missing_files = []
    failed_files = []

    for _, row in df.iterrows():
        filename = str(row["filename"]).strip()
        label = str(row["label_id"]).strip()

        audio_path = os.path.join(AUDIO_DIR, filename)

        if not os.path.exists(audio_path):
            missing_files.append(audio_path)
            continue

        try:
            features = extract_mfcc_delta_features(audio_path)
            X.append(features)
            y.append(label)
            filenames.append(filename)

        except Exception as error:
            failed_files.append((audio_path, str(error)))

    if missing_files:
        print("\nMissing audio files:")
        for path in missing_files:
            print("-", path)

    if failed_files:
        print("\nFiles with extraction errors:")
        for path, error in failed_files:
            print(f"- {path}: {error}")

    return np.array(X), np.array(y), np.array(filenames)


# ============================================================
# Filtering
# ============================================================

def filter_classes_with_enough_samples(X, y, filenames, min_samples=2):
    """
    Remove classes that have fewer than min_samples examples.

    This is necessary for StratifiedKFold.
    Example:
    If P03 has only 1 example, it cannot be split into 2 folds.
    """

    class_counts = pd.Series(y).value_counts().sort_index()

    valid_classes = class_counts[class_counts >= min_samples].index
    removed_classes = class_counts[class_counts < min_samples]

    mask = pd.Series(y).isin(valid_classes).to_numpy()

    X_filtered = X[mask]
    y_filtered = y[mask]
    filenames_filtered = filenames[mask]

    return X_filtered, y_filtered, filenames_filtered, class_counts, removed_classes


# ============================================================
# Evaluation
# ============================================================

def run_cross_validation(X, y):
    """
    Evaluate several SVM configurations with StratifiedKFold.
    """

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    class_counts = pd.Series(y).value_counts()
    min_class_count = class_counts.min()

    n_splits = min(2, min_class_count)

    if n_splits < 2:
        raise ValueError(
            "StratifiedKFold impossible even after filtering. "
            "At least one remaining class has fewer than 2 examples."
        )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    configs = [
        {"kernel": "rbf", "C": 0.1, "gamma": "scale"},
        {"kernel": "rbf", "C": 1, "gamma": "scale"},
        {"kernel": "rbf", "C": 10, "gamma": "scale"},
        {"kernel": "rbf", "C": 100, "gamma": "scale"},
        {"kernel": "linear", "C": 1, "gamma": "scale"},
        {"kernel": "linear", "C": 10, "gamma": "scale"},
    ]

    all_results = []
    best_result = None

    for config in configs:
        fold_accuracies = []
        fold_macro_f1s = []
        fold_reports = []

        print("\n========================================")
        print("Testing configuration:", config)
        print("========================================")

        for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), start=1):
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y_encoded[train_idx]
            y_test = y_encoded[test_idx]

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(
                    kernel=config["kernel"],
                    C=config["C"],
                    gamma=config["gamma"],
                    class_weight="balanced"
                ))
            ])

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            fold_accuracies.append(accuracy)
            fold_macro_f1s.append(macro_f1)

            labels_present = sorted(np.unique(np.concatenate([y_test, y_pred])))
            target_names = label_encoder.inverse_transform(labels_present)

            report = classification_report(
                y_test,
                y_pred,
                labels=labels_present,
                target_names=target_names,
                zero_division=0
            )

            fold_reports.append((fold_id, report))

            print(f"Fold {fold_id}/{n_splits}")
            print(f"Accuracy : {accuracy:.4f}")
            print(f"Macro F1 : {macro_f1:.4f}")

        result = {
            "config": config,
            "accuracy_mean": float(np.mean(fold_accuracies)),
            "accuracy_std": float(np.std(fold_accuracies)),
            "macro_f1_mean": float(np.mean(fold_macro_f1s)),
            "macro_f1_std": float(np.std(fold_macro_f1s)),
            "reports": fold_reports,
        }

        all_results.append(result)

        print("\nConfiguration summary:")
        print(f"Accuracy mean : {result['accuracy_mean']:.4f}")
        print(f"Accuracy std  : {result['accuracy_std']:.4f}")
        print(f"Macro F1 mean : {result['macro_f1_mean']:.4f}")
        print(f"Macro F1 std  : {result['macro_f1_std']:.4f}")

        if best_result is None or result["macro_f1_mean"] > best_result["macro_f1_mean"]:
            best_result = result

    return all_results, best_result, label_encoder, n_splits


# ============================================================
# Final model
# ============================================================

def train_final_model(X, y, best_config):
    """
    Train final model on all evaluable samples.
    """

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=best_config["kernel"],
            C=best_config["C"],
            gamma=best_config["gamma"],
            class_weight="balanced"
        ))
    ])

    final_model.fit(X, y_encoded)

    return final_model, label_encoder


# ============================================================
# Save results
# ============================================================

def save_results(
    all_results,
    best_result,
    raw_class_counts,
    evaluated_class_counts,
    removed_classes,
    n_splits
):
    """
    Save experiment results in a readable text file.
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("Experiment: MFCC + Delta + Delta-Delta + SVM\n")
        f.write("================================================\n\n")

        f.write("Description:\n")
        f.write(
            "This experiment evaluates a classical acoustic baseline for "
            "whistled sentence classification. Features are based on MFCC, "
            "delta MFCC and delta-delta MFCC. The classifier is an SVM.\n\n"
        )

        f.write("Feature extraction:\n")
        f.write(f"- Sample rate: {SAMPLE_RATE} Hz\n")
        f.write(f"- Number of MFCC coefficients: {N_MFCC}\n")
        f.write("- Feature type: MFCC + delta + delta-delta\n")
        f.write("- Statistics: mean + standard deviation over time\n")
        f.write("- Feature dimension: 120\n\n")

        f.write("Cross-validation:\n")
        f.write(f"- StratifiedKFold with n_splits = {n_splits}\n\n")

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

        f.write("All tested configurations:\n")
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

        f.write("\n\nDetailed classification reports for best configuration:\n")
        for fold_id, report in best_result["reports"]:
            f.write("\n")
            f.write(f"Fold {fold_id}\n")
            f.write("----------------------------------------\n")
            f.write(report)
            f.write("\n")


# ============================================================
# Main
# ============================================================

def main():
    print("Loading dataset and extracting MFCC features...")

    X, y, filenames = load_dataset()

    print("\nRaw dataset:")
    print("Number of audios found:", len(X))
    print("Number of classes:", len(set(y)))
    print("\nRaw class distribution:")
    raw_class_counts = pd.Series(y).value_counts().sort_index()
    print(raw_class_counts)

    X_eval, y_eval, filenames_eval, raw_counts, removed_classes = filter_classes_with_enough_samples(
        X,
        y,
        filenames,
        min_samples=2
    )

    print("\nDataset used for evaluation:")
    print("Number of audios used:", len(X_eval))
    print("Number of classes:", len(set(y_eval)))
    print("\nEvaluated class distribution:")
    evaluated_class_counts = pd.Series(y_eval).value_counts().sort_index()
    print(evaluated_class_counts)

    print("\nRemoved classes with fewer than 2 samples:")
    print(removed_classes if len(removed_classes) > 0 else "None")

    all_results, best_result, _, n_splits = run_cross_validation(X_eval, y_eval)

    print("\n========================================")
    print("Best configuration")
    print("========================================")
    print(best_result["config"])
    print(f"Accuracy mean : {best_result['accuracy_mean']:.4f}")
    print(f"Accuracy std  : {best_result['accuracy_std']:.4f}")
    print(f"Macro F1 mean : {best_result['macro_f1_mean']:.4f}")
    print(f"Macro F1 std  : {best_result['macro_f1_std']:.4f}")

    final_model, label_encoder = train_final_model(
        X_eval,
        y_eval,
        best_result["config"]
    )

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    joblib.dump({
        "model": final_model,
        "label_encoder": label_encoder,
        "feature_type": "mfcc_delta_deltadelta_mean_std",
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
        n_splits=n_splits
    )

    print(f"\nResults saved: {RESULTS_PATH}")
    print(f"Final model saved: {MODEL_OUT}")


if __name__ == "__main__":
    main()