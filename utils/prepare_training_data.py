"""
utils/prepare_training_data.py
--------------------------------
Prepares a labeled CSV dataset for fine-tuning the emotion classifier.

Sources combined:
  1. GoEmotions (Google, 58k Reddit comments, 27 emotions → collapsed to 7)
  2. ISEAR dataset (International Survey On Emotion Antecedents And Reactions)
  3. DailyDialog (dialog emotion annotations)

You do NOT need to run this unless you want to fine-tune a custom model.
The pretrained j-hartmann/emotion-english-distilroberta-base is already
excellent for most use cases.

Usage:
    python utils/prepare_training_data.py --output data/

This will create:
    data/train.csv
    data/val.csv
    data/label_distribution.txt
"""

import os
import csv
import random
import argparse
from collections import Counter


# ── GoEmotions label mapping (27 → 7) ───────────────────────────────────────
# GoEmotions has 27 fine-grained labels. We collapse them to our 7.
GOEMOTIONS_MAP = {
    # joy
    "admiration":    "joy",
    "amusement":     "joy",
    "approval":      "joy",
    "caring":        "joy",
    "desire":        "joy",
    "excitement":    "joy",
    "gratitude":     "joy",
    "joy":           "joy",
    "love":          "joy",
    "optimism":      "joy",
    "pride":         "joy",
    "relief":        "joy",
    # surprise
    "surprise":      "surprise",
    "realization":   "surprise",
    "curiosity":     "surprise",
    # sadness
    "disappointment":"sadness",
    "grief":         "sadness",
    "remorse":       "sadness",
    "sadness":       "sadness",
    # anger
    "anger":         "anger",
    "annoyance":     "anger",
    "disapproval":   "anger",
    # fear
    "fear":          "fear",
    "nervousness":   "fear",
    # disgust
    "disgust":       "disgust",
    "embarrassment": "disgust",
    # neutral
    "neutral":       "neutral",
    "confusion":     "neutral",
}

TARGET_LABELS = {"joy", "surprise", "sadness", "anger", "fear", "disgust", "neutral"}


def load_goemotions(data_dir: str) -> list[tuple[str, str]]:
    """
    Load GoEmotions TSV files from https://github.com/google-research/google-research/tree/master/goemotions
    Expected files: train.tsv, dev.tsv, test.tsv

    Format: text \t comma_separated_label_ids \t comment_id
    Labels are defined in emotions.txt in the same directory.
    """
    samples = []
    label_file = os.path.join(data_dir, "emotions.txt")

    if not os.path.exists(label_file):
        print(f"[GoEmotions] emotions.txt not found at {label_file}. Skipping.")
        return samples

    with open(label_file) as f:
        id2label = [l.strip() for l in f.readlines()]

    for split in ("train.tsv", "dev.tsv", "test.tsv"):
        fpath = os.path.join(data_dir, split)
        if not os.path.exists(fpath):
            continue
        with open(fpath, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 2:
                    continue
                text = row[0].strip()
                label_ids = [int(x) for x in row[1].split(",") if x.strip().isdigit()]
                if not label_ids:
                    continue
                # Take the first label only (or majority vote if you prefer)
                fine_label = id2label[label_ids[0]] if label_ids[0] < len(id2label) else None
                if fine_label and fine_label in GOEMOTIONS_MAP:
                    coarse = GOEMOTIONS_MAP[fine_label]
                    if coarse in TARGET_LABELS:
                        samples.append((text, coarse))

    print(f"[GoEmotions] Loaded {len(samples)} samples")
    return samples


def load_isear(filepath: str) -> list[tuple[str, str]]:
    """
    Load ISEAR dataset from https://www.affective-sciences.org/home/research/materials-and-online-research/research-material/
    CSV with columns: Field1, ...text column..., emotion_label

    We look for columns named 'SIT' (situation text) and 'EMOT' (emotion number).
    ISEAR emotion IDs: 1=joy, 2=fear, 3=anger, 4=sadness, 5=disgust, 6=shame, 7=guilt
    """
    ISEAR_MAP = {
        "1": "joy",
        "2": "fear",
        "3": "anger",
        "4": "sadness",
        "5": "disgust",
        "6": "sadness",   # shame → sadness
        "7": "sadness",   # guilt → sadness
    }
    samples = []
    if not os.path.exists(filepath):
        print(f"[ISEAR] File not found: {filepath}. Skipping.")
        return samples

    with open(filepath, newline="", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text  = row.get("SIT", "").strip()
            emot  = row.get("EMOT", "").strip()
            if text and emot in ISEAR_MAP:
                samples.append((text, ISEAR_MAP[emot]))

    print(f"[ISEAR] Loaded {len(samples)} samples")
    return samples


def make_synthetic_samples() -> list[tuple[str, str]]:
    """
    A small set of hand-crafted, high-quality samples to seed each class.
    These ensure every class has at least some clean training signal.
    """
    return [
        # joy
        ("I just got promoted! This is the best day of my life!", "joy"),
        ("Thank you so much, this really made my day.", "joy"),
        ("I am so excited about the trip, I can barely sleep!", "joy"),
        ("We won! I cannot believe we actually won!", "joy"),
        ("This is wonderful news. I am so happy for you.", "joy"),
        # anger
        ("I cannot believe they did this again. I am absolutely furious.", "anger"),
        ("This is completely unacceptable and I demand an explanation.", "anger"),
        ("Why does nobody listen to me? It is so frustrating!", "anger"),
        ("They lied to me and I am done with this company.", "anger"),
        ("Every single time. I have had enough of this nonsense.", "anger"),
        # sadness
        ("I miss her so much. The house feels empty without her.", "sadness"),
        ("I do not know what to do. I feel completely hopeless.", "sadness"),
        ("Nobody came. I sat there alone all evening.", "sadness"),
        ("I tried so hard and it still was not enough.", "sadness"),
        ("I feel like everything is falling apart.", "sadness"),
        # fear
        ("I am terrified of what the test results might say.", "fear"),
        ("I heard a noise downstairs and now I am scared to move.", "fear"),
        ("What if I lose my job? I cannot stop worrying.", "fear"),
        ("I have never been this anxious before an interview.", "fear"),
        ("Something is wrong and I do not know what to do.", "fear"),
        # surprise
        ("Wait, you are serious? I had no idea this was happening!", "surprise"),
        ("Oh wow, I did not expect that at all!", "surprise"),
        ("That came out of nowhere. I am completely speechless.", "surprise"),
        ("No way. I cannot believe it is actually true.", "surprise"),
        ("I was shocked when they told me the news.", "surprise"),
        # disgust
        ("This is revolting. I want nothing to do with it.", "disgust"),
        ("The smell alone made me gag. Absolutely disgusting.", "disgust"),
        ("I cannot believe someone would do something so vile.", "disgust"),
        ("That is the most repulsive thing I have ever seen.", "disgust"),
        ("It makes me sick to even think about it.", "disgust"),
        # neutral
        ("Please hold while I transfer your call.", "neutral"),
        ("Your account balance is one hundred and forty dollars.", "neutral"),
        ("The meeting has been rescheduled to Thursday at two PM.", "neutral"),
        ("Thank you for calling. How can I assist you today?", "neutral"),
        ("I will need your order number to look that up.", "neutral"),
    ]


def build_dataset(
    goemotions_dir: str = None,
    isear_csv:      str = None,
    output_dir:     str = "data",
    val_split:      float = 0.15,
    max_per_class:  int = 5000,
    seed:           int = 42,
):
    random.seed(seed)
    all_samples: list[tuple[str, str]] = []

    # Load each source
    if goemotions_dir:
        all_samples += load_goemotions(goemotions_dir)
    if isear_csv:
        all_samples += load_isear(isear_csv)
    all_samples += make_synthetic_samples()

    # Deduplicate by text
    seen = set()
    deduped = []
    for text, label in all_samples:
        key = text.lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append((text, label))
    all_samples = deduped

    # Balance classes (cap at max_per_class)
    by_class: dict[str, list] = {l: [] for l in TARGET_LABELS}
    for text, label in all_samples:
        if label in by_class:
            by_class[label].append((text, label))

    balanced = []
    for label, items in by_class.items():
        random.shuffle(items)
        balanced += items[:max_per_class]

    random.shuffle(balanced)

    # Train/val split
    split = int(len(balanced) * (1 - val_split))
    train = balanced[:split]
    val   = balanced[split:]

    # Write CSVs
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    val_path   = os.path.join(output_dir, "val.csv")

    for path, data in [(train_path, train), (val_path, val)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for text, label in data:
                writer.writerow([text, label])

    # Stats
    dist_path = os.path.join(output_dir, "label_distribution.txt")
    train_counts = Counter(l for _, l in train)
    val_counts   = Counter(l for _, l in val)
    with open(dist_path, "w") as f:
        f.write("Label distribution\n==================\n")
        f.write(f"{'Label':<12} {'Train':>8} {'Val':>8}\n")
        for label in sorted(TARGET_LABELS):
            f.write(f"{label:<12} {train_counts[label]:>8} {val_counts[label]:>8}\n")
        f.write(f"\n{'TOTAL':<12} {len(train):>8} {len(val):>8}\n")

    print(f"\nDataset saved to {output_dir}/")
    print(f"  train: {len(train)} samples → {train_path}")
    print(f"  val:   {len(val)} samples → {val_path}")
    print(f"  distribution: {dict(train_counts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goemotions_dir", default=None, help="Path to GoEmotions TSV directory")
    parser.add_argument("--isear_csv",      default=None, help="Path to ISEAR CSV file")
    parser.add_argument("--output",         default="data", help="Output directory")
    parser.add_argument("--max_per_class",  type=int, default=5000)
    args = parser.parse_args()

    build_dataset(
        goemotions_dir=args.goemotions_dir,
        isear_csv=args.isear_csv,
        output_dir=args.output,
        max_per_class=args.max_per_class,
    )
