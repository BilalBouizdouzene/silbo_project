import os
import re
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_CSV = "data/metadata/labels.csv"
OUTPUT_CSV = "data/metadata/labels_clean.csv"
LABEL_MAPPING_CSV = "data/metadata/label_mapping.csv"
ISSUES_CSV = "data/metadata/label_issues_to_check.csv"


# ============================================================
# Normalization utilities
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize IPA phrase labels before assigning class IDs.

    The purpose of this function is not to change the linguistic
    content of the annotation. The goal is only to remove small
    inconsistencies that would create artificial classes.

    Example:
        "komjˈo" and "komiˈo" refer to the same word "comió"
        in our current annotation convention, so we normalize them.
    """

    text = str(text).strip()

    # Remove silence markers if they were accidentally extracted
    # from the phrase tier.
    text = re.sub(r"\bsil\b", " ", text)

    # Remove secondary stress mark to avoid artificial differences.
    text = text.replace("ˌ", "")

    # Remove punctuation if present.
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace(";", "")
    text = text.replace(":", "")

    # ------------------------------------------------------------
    # Known transcription variants in the current dataset
    # ------------------------------------------------------------

    # comió: keep the convention ko-mi-o
    text = text.replace("komjˈo", "komiˈo")

    # bebió: keep the convention be-bi-o
    text = text.replace("beβjˈo", "bebiˈo")

    # necesita: harmonize Castilian theta convention
    text = text.replace("nesesˈita", "neθesˈita")

    # guía: remove orthographic leftover if it appears
    text = text.replace("guía", "ɡuˈia")

    # extranjero: harmonize older/variant transcription
    text = text.replace("extɾaŋˈxeɾo", "ekstɾanˈxeɾo")

    # accidental annotation artifact found in one file
    text = text.replace("ɡɾˈandesn b", "ɡɾˈandes")

    # harmonize b / beta variants in current annotations
    text = text.replace("salβˈo", "salbˈo")

    # harmonize Silvia variant for class grouping
    text = text.replace("sˈilβja", "sˈilbja")

    # Normalize ASCII g to IPA ɡ only after known replacements.
    # This avoids having both g and ɡ in labels.
    text = text.replace("g", "ɡ")

    # Clean spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def make_sentence_key(text: str) -> str:
    """
    Build a stable sentence key from the normalized phrase.

    Important:
    We do not infer the label from the filename prefix anymore.
    The same prefix can correspond to different sentences in
    different subsets of the corpus.
    """

    return normalize_text(text).lower().strip()


def detect_known_issues(row) -> list:
    """
    Detect suspicious cases that should be checked manually
    in Praat/TextGrid.

    This file is useful for transparency: Jose can see that
    we are not hiding annotation problems, and that we track them.
    """

    filename = str(row["filename"])
    phrase = str(row["phrase"])

    issues = []

    if filename.endswith("_1_.wav"):
        issues.append("Filename contains an extra underscore before .wav")

    if "ɡɾˈandesn b" in phrase:
        issues.append("Suspicious trailing characters: 'n b'")

    if "nesesˈita" in phrase:
        issues.append("Variant 'nesesˈita' should be checked against convention 'neθesˈita'")

    if "extɾaŋˈxeɾo" in phrase:
        issues.append("Variant 'extɾaŋˈxeɾo' should be checked against convention 'ekstɾanˈxeɾo'")

    if "salβˈo" in phrase:
        issues.append("Variant 'salβˈo' harmonized to 'salbˈo' for class grouping")

    if "ˌ" in phrase:
        issues.append("Secondary stress mark removed during normalization")

    return issues


# ============================================================
# Main script
# ============================================================

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_columns = {"filename", "phrase"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"The input CSV must contain columns {required_columns}. "
            f"Missing columns: {missing_columns}"
        )

    # Keep original phrase for traceability
    df["filename"] = df["filename"].astype(str).str.strip()
    df["original_phrase"] = df["phrase"].astype(str).str.strip()

    # Detect issues before normalization
    issue_rows = []

    for _, row in df.iterrows():
        issues = detect_known_issues(row)

        for issue in issues:
            issue_rows.append({
                "filename": row["filename"],
                "original_phrase": row["original_phrase"],
                "issue": issue
            })

    issues_df = pd.DataFrame(issue_rows)

    # Normalize phrase labels
    df["phrase"] = df["original_phrase"].apply(normalize_text)

    # Remove empty phrases if any
    df = df[df["phrase"].str.len() > 0].copy()

    # Create stable grouping key based on actual phrase content
    df["sentence_key"] = df["phrase"].apply(make_sentence_key)

    # Assign label IDs by sorted sentence keys for reproducibility
    unique_sentence_keys = sorted(df["sentence_key"].unique())

    label_mapping = {
        sentence_key: f"P{idx + 1:02d}"
        for idx, sentence_key in enumerate(unique_sentence_keys)
    }

    df["label_id"] = df["sentence_key"].map(label_mapping)

    # Build mapping file for readability
    mapping_rows = []

    for sentence_key, label_id in label_mapping.items():
        subset = df[df["sentence_key"] == sentence_key]

        mapping_rows.append({
            "label_id": label_id,
            "phrase": subset["phrase"].iloc[0],
            "count": len(subset)
        })

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df = mapping_df.sort_values("label_id").reset_index(drop=True)

    # Final labels file used by experiments
    labels_clean_df = df[[
        "filename",
        "phrase",
        "label_id"
    ]].copy()

    # Save outputs
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    labels_clean_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8"
    )

    mapping_df.to_csv(
        LABEL_MAPPING_CSV,
        index=False,
        encoding="utf-8"
    )

    if not issues_df.empty:
        issues_df.to_csv(
            ISSUES_CSV,
            index=False,
            encoding="utf-8"
        )

    # Console report
    print("labels_clean.csv created successfully.")
    print("label_mapping.csv created successfully.")

    if not issues_df.empty:
        print("label_issues_to_check.csv created successfully.")
    else:
        print("No suspicious label issues detected.")

    print()
    print("Number of audio files:", len(labels_clean_df))
    print("Number of classes:", labels_clean_df["label_id"].nunique())

    print()
    print("Class distribution:")
    print(labels_clean_df["label_id"].value_counts().sort_index())

    print()
    print("Label mapping:")
    print(mapping_df.to_string(index=False))

    if not issues_df.empty:
        print()
        print("Issues to check manually:")
        print(issues_df.to_string(index=False))


if __name__ == "__main__":
    main()