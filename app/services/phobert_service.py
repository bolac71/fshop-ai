"""
PhoBERT multi-label moderation service.

Loads a fine-tuned vinai/phobert-base model from HuggingFace Hub.
The model is trained via training/train_phobert.ipynb on Colab/Kaggle.

Set env var PHOBERT_MODERATION_MODEL to your HF Hub repo ID after training.
While the model is not yet available, this service returns zero scores so the
decision engine falls back entirely to rule-based scoring.
"""
import os
import json
import logging

import torch

from app.models.schemas import ModerationLabel

logger = logging.getLogger(__name__)

LABEL_NAMES = ["toxic", "hate_speech"]
HF_MODEL_ID = os.getenv("PHOBERT_MODERATION_MODEL", "")


class PhoBERTModerationService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.thresholds: dict[str, float] = {label: 0.5 for label in LABEL_NAMES}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if HF_MODEL_ID:
            self._load_model()
        else:
            logger.warning(
                "PHOBERT_MODERATION_MODEL not set — PhoBERT service running in stub mode. "
                "All ML scores will be 0.0 until a trained model is configured."
            )

    def _load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            logger.info(f"Loading PhoBERT moderation model from: {HF_MODEL_ID}")
            self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
            self.model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
            self.model.to(self.device)
            self.model.eval()

            # Load per-label thresholds if saved alongside the model
            try:
                from huggingface_hub import hf_hub_download
                threshold_path = hf_hub_download(HF_MODEL_ID, "thresholds.json")
                with open(threshold_path) as f:
                    self.thresholds = json.load(f)
            except Exception:
                logger.info("No thresholds.json found in Hub repo, using defaults (0.5 per label)")

            logger.info("PhoBERT moderation model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load PhoBERT model: {e}")
            self.model = None
            self.tokenizer = None

    def predict(self, normalized_text: str) -> list[ModerationLabel]:
        if self.model is None or self.tokenizer is None:
            # Stub: return zero scores — rule-based score drives the decision
            return [ModerationLabel(label=lbl, score=0.0) for lbl in LABEL_NAMES]

        inputs = self.tokenizer(
            normalized_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            scores = torch.sigmoid(logits).squeeze().cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]

        return [
            ModerationLabel(label=lbl, score=round(float(s), 4))
            for lbl, s in zip(LABEL_NAMES, scores)
        ]
