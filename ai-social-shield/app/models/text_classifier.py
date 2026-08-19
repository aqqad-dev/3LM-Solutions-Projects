from pathlib import Path
import json
import os

from dotenv import load_dotenv
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


MODEL_DIR = resolve_project_path(os.environ["TEXT_MODEL_DIR"])
LABELS_PATH = resolve_project_path(os.environ["TEXT_LABELS_PATH"])

with LABELS_PATH.open(encoding="utf-8") as file:
    LABELS = json.load(file)

class TextModel:
    def __init__(self) -> None:
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Model weights were not found: {MODEL_DIR}"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )

        self.classifier = pipeline(
            task="zero-shot-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,  # CPU
        )

    def classify(self, text: str) -> dict[str, float]:
        result = self.classifier(
            text,
            candidate_labels=LABELS,
            multi_label=True,
        )

        scores = {label: 0.0 for label in LABELS}

        for label, score in zip(result["labels"], result["scores"]):
            scores[label] = round(score * 100, 2)

        return scores


text_model = TextModel()