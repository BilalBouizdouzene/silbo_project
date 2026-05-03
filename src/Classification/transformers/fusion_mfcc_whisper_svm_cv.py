import os
import warnings
import numpy as np
import pandas as pd
import librosa
import joblib
import torch

from tqdm import tqdm
from transformers import WhisperProcessor, WhisperModel

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ============================================================
# General configuration
# ============================================================

warnings.filterwarnings("ignore", category=UserWarning)

AUDIO_DIR = "data/raw"
LABELS_PATH = "data/metadata/labels_clean.csv"

RESULTS_DIR = "experiments/exp_04_fusion_mfcc_whisper_svm"
RESULTS_PATH = os.path.join(RESULTS_DIR, "results.txt")

FEATURES_OUT = "data/processed/embeddings/fusion_mfcc_whisper_embeddings.npz"
MODEL_OUT = "models/fusion_mfcc_whisper_svm/fusion_mfcc_whisper_svm.joblib"

WHISPER_MODEL_NAME = "openai/whisper-tiny"

SAMPLE_RATE = 16000
N_MFCC = 20
RANDOM_STATE = 42


# ============================================================
# Audio loading
# ============================================================

def load_audio(audio_path: str, sr: int = 16000) -> np.ndarray:
    """
    Load one audio file and resample it to 16 kHz.

    The same sampling rate is used for MFCC extraction and Whisper.
    """

    audio, _ = librosa.load(audio_path, sr=sr)
    return audio


# ============================================================
# MFCC feature extraction
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
    Total MFCC part      = 120
    """

    # Remove leading/trailing silence if possible
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

    features = np.concatenate([
        summarize(mfcc),
        summarize(delta),
        summarize(delta_delta)
    ])

    return features


# ============================================================
# Whisper embedding extraction
# ============================================================

def extract_whisper_embedding(audio, processor, model, device) -> np.ndarray:
    """
    Extract one fixed-size embedding from the Whisper encoder.

    Workflow:
    audio waveform
        -> Whisper processor
        -> log-mel input features
        -> Whisper encoder
        -> hidden states
        -> mean pooling over time

    Whisper tiny encoder dimension = 384.
    """

    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        outputs = model.encoder(input_features)

    hidden_states = outputs.last_hidden_state

    embedding = hidden_states.mean(dim=1).squeeze(0)

    return embedding.cpu().numpy()


# ============================================================
# Fusion feature construction
# ============================================================

def build_fusion_features():
    """
    Build one feature vector per audio file.

    Final vector:
    [MFCC + delta + delta-delta summary] + [Whisper encoder embedding]

    Dimensions:
    - MFCC dynamic features: 120
    - Whisper tiny embedding: 384
    - Total: 504
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device used:", device)

    print(f"Loading Whisper model: {WHISPER_MODEL_NAME}")
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
    model = WhisperModel.from_pretrained(WHISPER_MODEL_NAME)

    model.to(device)
    model.eval()

    X = []
    y = []
    filenames = []
    missing_files = []
    failed_files = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting MFCC + Whisper features"):
        filename = str(row["filename"]).strip()
        label = str(row["label_id"]).strip()

        audio_path = os.path.join(AUDIO_DIR, filename)

        if not os.path.exists(audio_path):
            missing_files.append(audio_path)
            continue

        try:
            audio = load_audio(audio_path, sr=SAMPLE_RATE)

            mfcc_features = extract_mfcc_delta_features(
                audio=audio,
                sr=SAMPLE_RATE
            )

            whisper_embedding = extract_whisper_embedding(
                audio=audio,
                processor=processor,
                model=model,
                device=device
            )

            fusion_vector = np.concatenate([
                mfcc_features,
                whisper_embedding
            ])

            X.append(fusion_vector)
            y.append(label)
            filenames.append(filename)

        except Exception as error:
            failed_files.append((audio_path, str(error)))

    if missing_files:
        print("\nMissing audio files:")
        for path in missing_files:
            print("-", path)

    if failed_files:
        print("\nFiles with feature extraction errors:")
        for path, error in failed_files:
            print(f"- {path}: {error}")

    return np.array(X), np.array(y), np.array(filenames)


# ============================================================
# Feature cache
# ============================================================

def load_or_build_features():
    """
    Load cached fusion features if available, otherwise compute them.

    Important:
    If labels_clean.csv changes, delete the old .npz file before running
    this script. The script prints the number of loaded samples to make
    old caches easy to detect.
    """

    os.makedirs(os.path.dirname(FEATURES_OUT), exist_ok=True)

    if os.path.exists(FEATURES_OUT):
        print("Loading existing fusion features:", FEATURES_OUT)

        data = np.load(FEATURES_OUT, allow_pickle=True)

        X = data["X"]
        y = data["y"]
        filenames = data["filenames"]

        print("Loaded fusion features:", len(X))
        return X, y, filenames

    print("No saved fusion features found. Extracting features...")

    X, y, filenames = build_fusion_features()

    np.savez(
        FEATURES_OUT,
        X=X,
        y=y,
        filenames=filenames
    )

    print("Fusion features saved:", FEATURES_OUT)

    return X, y, filenames


# ============================================================
# Dataset filtering
# ============================================================

def filter_classes_with_enough_samples(X, y, filenames, min_samples=2):
    """
    Remove classes with fewer than min_samples examples.

    StratifiedKFold requires at least one sample of each class in each fold.
    With n_splits = 2, every evaluated class must have at least 2 samples.
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
# Cross-validation
# ============================================================

def run_cross_validation(X, y):
    """
    Evaluate several SVM configurations using StratifiedKFold.

    The best configuration is selected according to mean Macro F1.
    """

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    min_class_count = pd.Series(y).value_counts().min()
    n_splits = min(2, min_class_count)

    if n_splits < 2:
        raise ValueError(
            "StratifiedKFold is impossible even after filtering. "
            "At least one remaining class has fewer than 2 samples."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    svm_configs = [
        {"kernel": "rbf", "C": 0.1, "gamma": "scale"},
        {"kernel": "rbf", "C": 1, "gamma": "scale"},
        {"kernel": "rbf", "C": 10, "gamma": "scale"},
        {"kernel": "rbf", "C": 100, "gamma": "scale"},
        {"kernel": "linear", "C": 1, "gamma": "scale"},
        {"kernel": "linear", "C": 10, "gamma": "scale"},
    ]

    all_results = []
    best_result = None

    for config in svm_configs:
        fold_accuracies = []
        fold_macro_f1s = []
        fold_reports = []

        print("\n========================================")
        print("Testing configuration:", config)
        print("========================================")

        for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y_encoded), start=1):
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y_encoded[train_idx]
            y_test = y_encoded[test_idx]

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(
                    kernel=config["kernel"],
                    C=config["C"],
                    gamma=config["gamma"],
                    class_weight="balanced"
                ))
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0
            )

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
# Final model training
# ============================================================

def train_final_model(X, y, best_config):
    """
    Train final SVM model on all evaluable samples.
    """

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=best_config["kernel"],
            C=best_config["C"],
            gamma=best_config["gamma"],
            class_weight="balanced"
        ))
    ])

    final_pipeline.fit(X, y_encoded)

    return final_pipeline, label_encoder


# ============================================================
# Save results
# ============================================================

def save_results(
    all_results,
    best_result,
    raw_class_counts,
    evaluated_class_counts,
    removed_classes,
    n_splits,
    feature_dim
):
    """
    Save a transparent and readable results file.
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("Experiment: Fusion MFCC + Delta + Delta-Delta + Whisper Encoder + SVM\n")
        f.write("=====================================================================\n\n")

        f.write("Description:\n")
        f.write(
            "This experiment combines classical acoustic features with Whisper encoder "
            "embeddings. The acoustic part uses MFCC, delta MFCC and delta-delta MFCC "
            "summarized with mean and standard deviation. The neural part uses the "
            "Whisper encoder with mean pooling over time. The final classifier is an SVM.\n\n"
        )

        f.write("Feature extraction:\n")
        f.write(f"- Sample rate: {SAMPLE_RATE} Hz\n")
        f.write(f"- Number of MFCC coefficients: {N_MFCC}\n")
        f.write("- Acoustic features: MFCC + delta + delta-delta, mean + std\n")
        f.write("- Acoustic feature dimension: 120\n")
        f.write(f"- Whisper model: {WHISPER_MODEL_NAME}\n")
        f.write("- Whisper embedding dimension: 384\n")
        f.write(f"- Total feature dimension: {feature_dim}\n\n")

        f.write("Cross-validation:\n")
        f.write(f"- StratifiedKFold with n_splits = {n_splits}\n")
        f.write("- Classes with fewer than 2 samples are excluded from evaluation\n\n")

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

        f.write("\n\nDetailed classification reports for the best configuration:\n")
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
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    print("Loading or extracting fusion features...")

    X, y, filenames = load_or_build_features()

    print("\nRaw dataset:")
    print("Number of audios found:", len(X))
    print("Feature dimension:", X.shape[1])
    print("Number of classes:", len(set(y)))

    raw_class_counts = pd.Series(y).value_counts().sort_index()

    print("\nRaw class distribution:")
    print(raw_class_counts)

    X_eval, y_eval, filenames_eval, raw_counts, removed_classes = filter_classes_with_enough_samples(
        X,
        y,
        filenames,
        min_samples=2
    )

    evaluated_class_counts = pd.Series(y_eval).value_counts().sort_index()

    print("\nDataset used for evaluation:")
    print("Number of audios used:", len(X_eval))
    print("Number of classes:", len(set(y_eval)))

    print("\nEvaluated class distribution:")
    print(evaluated_class_counts)

    print("\nRemoved classes with fewer than 2 samples:")
    print(removed_classes if len(removed_classes) > 0 else "None")

    all_results, best_result, _, n_splits = run_cross_validation(
        X_eval,
        y_eval
    )

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

    joblib.dump({
        "pipeline": final_model,
        "label_encoder": label_encoder,
        "feature_type": "mfcc_delta_deltadelta_plus_whisper_encoder",
        "whisper_model": WHISPER_MODEL_NAME,
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
        feature_dim=X.shape[1]
    )

    print(f"\nResults saved: {RESULTS_PATH}")
    print(f"Final model saved: {MODEL_OUT}")


if __name__ == "__main__":
    main()