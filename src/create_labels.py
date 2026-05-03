import os
import csv
import re

# chemins
AUDIO_DIR = "data/raw"
ANNOT_DIR = "data/annotations"
OUTPUT_FILE = os.path.join("data", "metadata", "labels.csv")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


def read_textgrid(path):
    """Read a TextGrid file and return its lines as text."""
    for encoding in ("utf-16", "utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read().splitlines()
        except UnicodeError:
            continue
    raise ValueError(f"Could not read TextGrid file: {path}")


def parse_textgrid_tier(tier_lines):
    result = {
        "name": "",
        "texts": [],
    }
    for line in tier_lines:
        if line.startswith("name ="):
            match = re.match(r'name\s*=\s*"(.*)"', line)
            if match:
                result["name"] = match.group(1).strip()
        elif line.startswith("text ="):
            match = re.match(r'text\s*=\s*"(.*)"', line)
            if match:
                result["texts"].append(match.group(1).strip())
    return result


def parse_textgrid_phrase(lines):
    """Parse a TextGrid and return the best candidate phrase from its tiers."""
    trimmed = [line.strip() for line in lines if line.strip()]

    tiers = []
    current_tier = []
    for line in trimmed:
        if line.startswith("item ["):
            if current_tier:
                tiers.append(parse_textgrid_tier(current_tier))
            current_tier = [line]
        elif current_tier:
            current_tier.append(line)
    if current_tier:
        tiers.append(parse_textgrid_tier(current_tier))

    if not tiers:
        return ""

    for tier in tiers:
        if tier["name"] and "phrase" in tier["name"].lower():
            return " ".join(t for t in tier["texts"] if t and not t.isspace()).strip()

    best_tier = max(
        tiers,
        key=lambda t: sum(1 for text in t["texts"] if text and not text.isspace()),
        default={"texts": []},
    )
    return " ".join(t for t in best_tier["texts"] if t and not t.isspace()).strip()


def build_labels():
    rows = []

    for filename in os.listdir(ANNOT_DIR):
        if filename.lower().endswith(".textgrid"):
            tg_path = os.path.join(ANNOT_DIR, filename)
            audio_name = filename[:-9] + ".wav"

            lines = read_textgrid(tg_path)
            phrase = parse_textgrid_phrase(lines)

            rows.append([audio_name, phrase])

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "phrase"])
        writer.writerows(rows)

    print("labels.csv créé avec succès !")


if __name__ == "__main__":
    build_labels()