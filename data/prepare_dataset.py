"""
Real-Time Sentiment Analysis - dataset preparation.

Source: Kaggle "Sentiment140" (kazanova) - 1.6 million real tweets,
labeled positive/negative via emoticon heuristics (a well-known
standard benchmark for short-text sentiment classification).
Columns (no header): target, id, date, flag, user, text
target: 0 = negative, 4 = positive

This script cleans the raw text (lowercase, strip URLs/@mentions/
punctuation noise) and draws a balanced random sample (real tweets,
not synthetic) small enough to train quickly on a CPU, saving it as
data/sentiment_dataset.csv (text,label) for model/train_model.py.

Run:
    python prepare_dataset.py
"""

import csv
import os
import random
import re

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "kaggle_reference", "training.1600000.processed.noemoticon.csv")
OUT_PATH = os.path.join(BASE_DIR, "sentiment_dataset.csv")

SAMPLES_PER_CLASS = 40000

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NON_ALPHA_RE = re.compile(r"[^a-z\s']")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text):
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def main():
    pos_rows = []
    neg_rows = []

    with open(RAW_PATH, encoding="latin-1", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            target, _id, _date, _flag, _user, text = row
            cleaned = clean_text(text)
            if len(cleaned.split()) < 2:
                continue
            if target == "0" and len(neg_rows) < SAMPLES_PER_CLASS * 3:
                neg_rows.append(cleaned)
            elif target == "4" and len(pos_rows) < SAMPLES_PER_CLASS * 3:
                pos_rows.append(cleaned)
            if len(pos_rows) >= SAMPLES_PER_CLASS * 3 and len(neg_rows) >= SAMPLES_PER_CLASS * 3:
                break

    random.shuffle(pos_rows)
    random.shuffle(neg_rows)
    pos_sample = pos_rows[:SAMPLES_PER_CLASS]
    neg_sample = neg_rows[:SAMPLES_PER_CLASS]

    rows = [(t, "positive") for t in pos_sample] + [(t, "negative") for t in neg_sample]
    random.shuffle(rows)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows ({len(pos_sample)} positive, {len(neg_sample)} negative) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
