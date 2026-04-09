import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
import re
from app.core.config import SENTIMENT_MODEL_NAME

# --- CONFIG NGƯỠNG (THRESHOLDS) ---
# Nếu model chắc chắn > 80% là tiêu cực -> Đánh dấu Toxic
NEGATIVE_THRESHOLD = 0.70 
# Nếu độ tin cậy < 65% (Lưng chừng) -> Cần LLM kiểm tra lại
UNCERTAINTY_THRESHOLD = 0.60

class SentimentService:
    def __init__(self):
        print("🚀 Loading multilingual Sentiment Model (Vietnamese-friendly)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model_name = SENTIMENT_MODEL_NAME

        self.bad_words = [
            r"\bfuck\b", r"\bshit\b", r"\bbitch\b", r"\basshole\b", 
            r"\bdick\b", r"\bcunt\b", r"\bpussy\b", r"\bastard\b",
            r"\bidiot\b", r"\bstupid\b", r"\bkill yourself\b", r"\bdie\b",
            r"\bdeo\b", r"\bđeo\b", r"\bdit\b", r"\bđịt\b", r"\bdu me\b", r"\bđụ mẹ\b",
            r"\bvkl\b", r"\bvc\b", r"\bcc\b", r"\bloz\b", r"\blon\b", r"\blồn\b",
            r"\bngu\b", r"\boc cho\b", r"\bóc chó\b", r"\bkhung\b", r"\bđần\b", r"\bdm\b"
        ]
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            print("✅ Sentiment Service Ready!")
        except Exception as e:
            print(f"❌ Error loading Sentiment Model: {e}")
            raise

    def _check_keywords(self, text: str):
        """Kiểm tra nhanh bằng Regex"""
        text_lower = text.lower()
        for pattern in self.bad_words:
            if re.search(pattern, text_lower):
                return True, pattern.replace(r"\b", "")
        return False, None

    def analyze(self, text: str):
        """
        Phan tich sentiment cho noi dung tieng Viet/da ngon ngu.
        Mac dinh map labels: 0 -> Negative, 1 -> Neutral, 2 -> Positive.
        """
        # BƯỚC 1: HARD FILTER
        is_bad_word, word_found = self._check_keywords(text)
        if is_bad_word:
            return {
                "label": "NEGATIVE",
                "score": 0.99, # Chắc chắn 99%
                "probabilities": {"neg": 0.99, "neu": 0.01, "pos": 0.0},
                "is_toxic": True,
                "requires_llm_check": False,
                "reason": f"Contains profanity/toxic keyword: {word_found}"
            }
        # BƯỚC 2: AI SENTIMENT MODEL
        encoded_input = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            output = self.model(**encoded_input)
        
        scores = output.logits[0].cpu().detach().numpy()
        probs = softmax(scores) # [Negative, Neutral, Positive]
        
        labels = ["negative", "neutral", "positive"]
        id2label = getattr(self.model.config, "id2label", None)
        if isinstance(id2label, dict) and len(id2label) >= 3:
            normalized = []
            for idx in range(len(probs)):
                raw = str(id2label.get(idx, "")).lower()
                if any(k in raw for k in ["neg", "1 star", "star_1", "label_0"]):
                    normalized.append("negative")
                elif any(k in raw for k in ["neu", "3 star", "star_3", "label_1"]):
                    normalized.append("neutral")
                elif any(k in raw for k in ["pos", "5 star", "star_5", "label_2"]):
                    normalized.append("positive")
                else:
                    normalized.append("")
            if {"negative", "neutral", "positive"}.issubset(set(normalized)):
                labels = normalized
        
        ranking = np.argsort(probs)
        top_label_idx = ranking[-1]
        
        top_label = labels[top_label_idx]
        confidence = float(probs[top_label_idx])
        
        is_toxic = False
        requires_review = False
        reason = "AI Model Classification"

        # BƯỚC 3: LOGIC NGHIỆP VỤ
        if top_label == "negative":
            if confidence > NEGATIVE_THRESHOLD:
                is_toxic = True
                reason = "High negative sentiment"
            else:
                requires_review = True
                
        elif top_label == "neutral":
            if confidence < 0.6: 
                requires_review = True
        
        return {
            "label": top_label.upper(),
            "score": confidence,
            "probabilities": {
                "neg": float(probs[0]),
                "neu": float(probs[1]),
                "pos": float(probs[2])
            },
            "is_toxic": is_toxic,
            "requires_llm_check": requires_review,
            "reason": reason
        }