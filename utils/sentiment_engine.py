"""
Real-Time Sentiment Analysis - core engine.

Wraps the trained TF-IDF + Logistic Regression classifier and adds the
lightweight "emotion-aware agent" layer described in the project
abstract: a small set of autonomous agents that each observe the same
detected emotional state and respond according to their own role and
rule set.

NOTE on scope: the abstract describes "SentioNet", a full multi-agent
world with per-agent beliefs/goals and inter-agent negotiation. This
prototype implements the emotion-detection core with real ML plus a
transparent rule-based agent-reaction layer - it does NOT implement
autonomous inter-agent negotiation or learned agent policies. See
README.md for exactly what is and isn't implemented.
"""

import os
import re

import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "model", "sentiment_model.pkl")

NEUTRAL_BAND = 0.12  # |p(pos) - 0.5| below this is reported as Neutral

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NON_ALPHA_RE = re.compile(r"[^a-z\s']")
MULTI_SPACE_RE = re.compile(r"\s+")

_vectorizer = None
_model = None


def _load():
    global _vectorizer, _model
    if _vectorizer is None or _model is None:
        if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH)):
            raise FileNotFoundError(
                "Trained model not found. Run 'python data/prepare_dataset.py' then "
                "'python model/train_model.py' first."
            )
        _vectorizer = joblib.load(VECTORIZER_PATH)
        _model = joblib.load(MODEL_PATH)
    return _vectorizer, _model


def clean_text(text):
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    return MULTI_SPACE_RE.sub(" ", text).strip()


def analyze(text):
    """Classifies text and returns sentiment, confidence, and agent reactions."""
    vectorizer, model = _load()

    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Please enter some text with at least one word.")

    X = vectorizer.transform([cleaned])
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    p_positive = float(proba[classes.index("positive")])

    if abs(p_positive - 0.5) < NEUTRAL_BAND:
        sentiment = "Neutral"
    elif p_positive > 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    confidence = round(max(p_positive, 1 - p_positive) * 100, 1)
    polarity = round((p_positive - 0.5) * 200, 1)  # -100 (very negative) .. +100 (very positive)

    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "polarity": polarity,
        "p_positive": round(p_positive * 100, 1),
        "agents": agent_reactions(sentiment, confidence),
        "influential_words": influential_words(cleaned, vectorizer, model),
    }


def influential_words(cleaned_text, vectorizer, model):
    """Returns the words in this text that pushed the prediction hardest,
    using the linear model's own coefficients - the interpretability the
    abstract calls for ('rule-based logic...to ensure interpretability')."""
    classes = list(model.classes_)
    pos_index = classes.index("positive")
    sign = 1 if pos_index == 1 else -1

    vocab = vectorizer.vocabulary_
    coefs = model.coef_[0]

    scored = []
    seen = set()
    for token in cleaned_text.split():
        if token in seen:
            continue
        seen.add(token)
        idx = vocab.get(token)
        if idx is not None:
            weight = float(coefs[idx]) * sign
            if abs(weight) > 0.05:
                scored.append({"word": token, "weight": round(weight, 3)})

    scored.sort(key=lambda item: abs(item["weight"]), reverse=True)
    return scored[:6]


def agent_reactions(sentiment, confidence):
    """Three rule-based agents, each with its own role, reacting to the same
    detected emotional state - the 'emotion-aware agent behaviour' layer."""
    low_confidence = confidence < 65

    if sentiment == "Positive":
        return [
            {"agent": "Support Agent", "role": "Wellbeing", "action": "Acknowledge and reinforce",
             "message": "That sounds genuinely positive — worth noting what went well here."},
            {"agent": "Engagement Agent", "role": "Interaction", "action": "Continue the thread",
             "message": "Good moment to keep the conversation going or ask a follow-up."},
            {"agent": "Escalation Agent", "role": "Risk monitoring", "action": "Stand down",
             "message": "No concern detected. No escalation needed."},
        ]
    if sentiment == "Negative":
        return [
            {"agent": "Support Agent", "role": "Wellbeing", "action": "Respond empathetically",
             "message": "Negative sentiment detected — lead with acknowledgement, not solutions."},
            {"agent": "Engagement Agent", "role": "Interaction", "action": "Slow the pace",
             "message": "Avoid upbeat or promotional replies right now; keep responses short and calm."},
            {"agent": "Escalation Agent", "role": "Risk monitoring",
             "action": "Flag for review" if confidence >= 80 else "Monitor",
             "message": ("Strong negative signal — flag this for a human to review."
                         if confidence >= 80 else
                         "Mild negative signal — keep monitoring before escalating.")},
        ]
    return [
        {"agent": "Support Agent", "role": "Wellbeing", "action": "Stay neutral",
         "message": "No clear emotional signal — respond factually rather than emotionally."},
        {"agent": "Engagement Agent", "role": "Interaction", "action": "Ask a clarifying question",
         "message": "Sentiment is ambiguous — a clarifying question would resolve intent."},
        {"agent": "Escalation Agent", "role": "Risk monitoring", "action": "Stand down",
         "message": ("Low model confidence — treat this classification with caution."
                     if low_confidence else "No concern detected. No escalation needed.")},
    ]
