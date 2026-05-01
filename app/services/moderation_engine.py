import os
from dataclasses import dataclass

from app.models.schemas import ModerationLabel


RULE_WEIGHT = float(os.getenv("MODERATION_RULE_WEIGHT", "0.30"))
ML_WEIGHT = 1.0 - RULE_WEIGHT
FLAG_THRESHOLD = float(os.getenv("MODERATION_FLAG_THRESHOLD", "0.35"))
HIGH_PRIORITY_THRESHOLD = float(os.getenv("MODERATION_HIGH_PRIORITY_THRESHOLD", "0.70"))


@dataclass
class ModerationDecision:
    decision: str    # "approved" | "flagged"
    priority: str    # "NORMAL" | "HIGH"
    final_score: float
    confidence: float


class ModerationEngine:
    def decide(
        self,
        rule_score: float,
        ml_labels: list[ModerationLabel],
    ) -> ModerationDecision:
        if ml_labels:
            ml_score = max(label.score for label in ml_labels)
            # Higher confidence when rule and ML agree
            rule_agreement = 1.0 - abs(rule_score - ml_score)
            confidence = 0.5 + 0.5 * rule_agreement
        else:
            ml_score = 0.0
            confidence = 0.5

        final_score = RULE_WEIGHT * rule_score + ML_WEIGHT * ml_score

        if final_score >= FLAG_THRESHOLD:
            decision = "flagged"
            priority = "HIGH" if final_score >= HIGH_PRIORITY_THRESHOLD else "NORMAL"
        else:
            decision = "approved"
            priority = "NORMAL"

        return ModerationDecision(
            decision=decision,
            priority=priority,
            final_score=round(final_score, 4),
            confidence=round(confidence, 4),
        )
