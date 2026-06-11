"""
BERT-based delay root-cause classifier.

Because the raw Delhivery dataset has no free-text delay notes, this module:
  1. Generates synthetic delay-reason text from structured features
     (delay_ratio, time_of_day, route_type, distance).
  2. Fine-tunes bert-base-uncased on those labels.
  3. Exposes a predict() function for inference on new trip records.

Delay categories:
  0 = traffic_congestion
  1 = weather
  2 = hub_congestion
  3 = vehicle_breakdown
  4 = last_mile_failure
  5 = on_time (no significant delay)

The BERT classifier's output enriches the graph with a semantic "why delayed"
signal beyond raw delay_ratio, enabling targeted interventions per corridor.
"""

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

LABEL_NAMES = [
    "traffic_congestion",
    "weather",
    "hub_congestion",
    "vehicle_breakdown",
    "last_mile_failure",
    "on_time",
]
N_LABELS = len(LABEL_NAMES)
MODEL_DIR = Path("data/processed/delay_classifier")


# ---------------------------------------------------------------------------
# 1.  Synthetic text generation
# ---------------------------------------------------------------------------

_TEMPLATES = {
    "traffic_congestion": [
        "Delivery delayed due to heavy traffic on NH-{n}.",
        "Severe congestion on the highway approaching {city} caused delay.",
        "Traffic jam on the route to {city} extended trip duration.",
    ],
    "weather": [
        "Heavy rainfall in {city} region delayed shipment.",
        "Fog and low visibility on route caused significant slowdown.",
        "Cyclonic weather near {city} grounded vehicles for {h} hours.",
    ],
    "hub_congestion": [
        "Processing backlog at {city} sorting center slowed outbound dispatch.",
        "Hub at {city} operating at 130% capacity, dwell time elevated.",
        "Inbound surge at {city} gateway hub caused {h}-hour delay.",
    ],
    "vehicle_breakdown": [
        "Vehicle breakdown on route near {city} required replacement truck.",
        "Tyre puncture on NH-{n} near {city} caused unplanned stoppage.",
        "Engine failure of delivery vehicle delayed shipment by {h} hours.",
    ],
    "last_mile_failure": [
        "Recipient unavailable at delivery address in {city}.",
        "Address mismatch caused re-delivery attempt in {city}.",
        "Last-mile rider could not locate address in {city} locality.",
    ],
    "on_time": [
        "Shipment delivered within estimated time to {city}.",
        "No significant delays. Route to {city} clear.",
        "Trip completed ahead of OSRM estimate.",
    ],
}

_CITIES = [
    "Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
]


def _synthetic_text(label: str, rng: random.Random) -> str:
    template = rng.choice(_TEMPLATES[label])
    city = rng.choice(_CITIES)
    n = rng.randint(1, 99)
    h = rng.randint(1, 6)
    return template.format(city=city, n=n, h=h)


def _label_from_features(row: pd.Series, rng: random.Random) -> int:
    """Rule-based heuristic to assign a delay label from structured features."""
    dr = row.get("delay_ratio", 1.0)
    tod = str(row.get("time_of_day", "")).lower()
    dist = row.get("osrm_distance", 500)

    if dr < 1.10:
        return 5  # on_time

    if tod in ("morning", "evening") and dr > 1.3:
        return rng.choices([0, 2], weights=[0.6, 0.4])[0]  # traffic or hub
    if dist > 800 and dr > 1.4:
        return rng.choices([1, 3], weights=[0.5, 0.5])[0]  # weather or breakdown
    if dist < 200 and dr > 1.25:
        return 4  # last-mile
    if dr > 1.5:
        return rng.choices([0, 1, 2, 3], weights=[0.4, 0.2, 0.3, 0.1])[0]
    return rng.choices([0, 2], weights=[0.55, 0.45])[0]


def generate_synthetic_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    texts, labels = [], []
    for _, row in df.iterrows():
        label_idx = _label_from_features(row, rng)
        label_name = LABEL_NAMES[label_idx]
        text = _synthetic_text(label_name, rng)
        texts.append(text)
        labels.append(label_idx)
    return pd.DataFrame({"text": texts, "label": labels, "label_name": [LABEL_NAMES[l] for l in labels]})


# ---------------------------------------------------------------------------
# 2.  Fine-tuning
# ---------------------------------------------------------------------------

def fine_tune(
    train_df: pd.DataFrame,
    epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 64,
    save_dir: str | Path = MODEL_DIR,
) -> None:
    """Fine-tune bert-base-uncased on the synthetic delay dataset."""
    try:
        import torch
        from transformers import (
            BertForSequenceClassification,
            BertTokenizer,
            Trainer,
            TrainingArguments,
        )
        from datasets import Dataset
    except ImportError as e:
        log.error(f"Transformers / datasets not installed: {e}")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    dataset = Dataset.from_pandas(train_df[["text", "label"]])
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    split = dataset.train_test_split(test_size=0.15, seed=42)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=N_LABELS
    )

    args = TrainingArguments(
        output_dir=str(save_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=str(save_dir / "logs"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )
    trainer.train()
    trainer.save_model(str(save_dir / "best_model"))
    tokenizer.save_pretrained(str(save_dir / "best_model"))
    log.info(f"Model saved → {save_dir / 'best_model'}")


# ---------------------------------------------------------------------------
# 3.  Inference
# ---------------------------------------------------------------------------

def load_classifier(model_dir: str | Path = MODEL_DIR):
    try:
        from transformers import pipeline as hf_pipeline
        classifier = hf_pipeline(
            "text-classification",
            model=str(Path(model_dir) / "best_model"),
            return_all_scores=False,
        )
        return classifier
    except Exception as e:
        log.warning(f"Could not load classifier: {e}. Using rule-based fallback.")
        return None


def predict(texts: list[str], classifier=None) -> list[str]:
    if classifier is None:
        return ["unknown"] * len(texts)
    results = classifier(texts, truncation=True, max_length=64)
    return [r["label"] for r in results]


def enrich_with_delay_reason(df: pd.DataFrame, classifier=None, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic delay texts then predict (or assign) delay reasons."""
    synth = generate_synthetic_dataset(df, seed=seed)
    if classifier is not None:
        synth["predicted_label"] = predict(synth["text"].tolist(), classifier)
    else:
        synth["predicted_label"] = synth["label_name"]
    df = df.copy()
    df["delay_reason_text"] = synth["text"].values
    df["delay_reason"] = synth["predicted_label"].values
    return df


if __name__ == "__main__":
    sample_df = pd.DataFrame({
        "delay_ratio": [1.05, 1.45, 1.80, 2.10, 1.15],
        "time_of_day": ["morning", "evening", "night", "afternoon", "morning"],
        "osrm_distance": [120, 650, 900, 180, 400],
    })
    synth = generate_synthetic_dataset(sample_df)
    print(synth)
