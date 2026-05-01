import re
import unicodedata
from dataclasses import dataclass, field


# Vietnamese toxic words and patterns for rule-based scoring
_TOXIC_PATTERNS_VI = [
    r"\b(đ[eéèẹê][oó]|đ[ụu][._ ]*m[eẹê]|đụ|địt|đ[ị][tú]|đmm?|vcl|vkl|[cđ][l]\b)",
    r"\b(lồn|l[o0][3ô]n|ngu|óc\s*ch[oó]|[kc]h[uù]ng|đần|đ[aầã]n\s*[oô]n|khùng)\b",
    r"\b(mả\s*m[eẹê]|m[eẹê]\s*m[àáạảã]y|th[ằắặẳẵ][nng]\s*ch[oó])\b",
    r"\b(cc|cu|cặc|[cđ][ặa][cg]|buồi|b[uù][oô]i)\b",
    r"\b(chó\s*ch[eẹê]t|ch[eẹê]t\s*đi|đ[eẹê][oó]\s*m[àáạảã]y)\b",
]
_TOXIC_PATTERNS_EN = [
    r"\b(f[u*@#!]+ck|sh[i!1]+t|b[i!1]+tch|[a@]ssh[o0]le|d[i!1]ck|c[u*]nt|wh[o0]re|f[a@]gg[o0]t)\b",
    r"\b(bastard|motherfucker|mf\b|stfu|gtfo|kys)\b",
]
_SPAM_PATTERNS = [
    r"(https?://\S+|www\.\S+)",           # URLs
    r"(\b0[0-9]{9,10}\b|\+84[0-9]{9})",  # Vietnamese phone numbers
    r"(zalo|telegram|fb\.com|tele\s*gram)\s*[\:：]?\s*\S+",
    r"(inbox|ib|liên\s*hệ|nhắn\s*tin)\s+(t[oô]i|mình|mk|m)",
]
_LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a",
    "@": "a", "$": "s", "!": "i", "+": "t",
})


@dataclass
class ProcessedText:
    original: str
    normalized: str
    rule_score: float
    signals: dict = field(default_factory=dict)
    matched_patterns: list = field(default_factory=list)


class TextPreprocessor:
    def __init__(self):
        self._toxic_vi = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in _TOXIC_PATTERNS_VI]
        self._toxic_en = [re.compile(p, re.IGNORECASE) for p in _TOXIC_PATTERNS_EN]
        self._spam = [re.compile(p, re.IGNORECASE) for p in _SPAM_PATTERNS]

    def preprocess(self, raw_text: str) -> ProcessedText:
        signals: dict = {}
        matched: list = []

        # 1. Unicode normalization (NFC for Vietnamese)
        normalized = unicodedata.normalize("NFC", raw_text).strip()

        # 2. Detect and normalize leet speak
        leet_normalized = normalized.translate(_LEET_MAP)
        signals["leet_detected"] = leet_normalized.lower() != normalized.lower()

        # 3. Repetition abuse detection (aaaa, !!!!, ...)
        repetition_re = re.compile(r"(.)\1{3,}")
        signals["repetition_abuse"] = bool(repetition_re.search(normalized))
        # Normalize repetitions for matching
        clean = repetition_re.sub(r"\1\1", leet_normalized)

        # 4. All-caps ratio
        alpha_chars = [c for c in normalized if c.isalpha()]
        caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)
        signals["caps_ratio"] = round(caps_ratio, 3)

        # 5. Spam signals
        url_count = sum(len(p.findall(normalized)) for p in self._spam[:1])
        phone_count = sum(len(p.findall(normalized)) for p in self._spam[1:2])
        contact_count = sum(len(p.findall(normalized)) for p in self._spam[2:])
        signals["url_count"] = url_count
        signals["phone_count"] = phone_count
        signals["contact_spam"] = contact_count > 0
        if url_count or phone_count or contact_count:
            matched.append("spam_signal")

        # 6. Toxic word matching (on both original & leet-normalized)
        for pat in self._toxic_vi + self._toxic_en:
            for text_variant in (normalized, clean):
                m = pat.search(text_variant)
                if m:
                    matched.append(m.group(0).lower())

        matched = list(dict.fromkeys(matched))  # deduplicate, preserve order
        signals["toxic_word_count"] = len([m for m in matched if m != "spam_signal"])

        # 7. Compute rule_score (0.0 – 1.0)
        rule_score = self._compute_rule_score(signals, matched)

        return ProcessedText(
            original=raw_text,
            normalized=normalized,
            rule_score=round(rule_score, 4),
            signals=signals,
            matched_patterns=matched,
        )

    def _compute_rule_score(self, signals: dict, matched: list) -> float:
        score = 0.0

        toxic_count = signals.get("toxic_word_count", 0)
        if toxic_count >= 3:
            score += 0.70
        elif toxic_count == 2:
            score += 0.55
        elif toxic_count == 1:
            score += 0.40

        # Spam signals add on top
        if signals.get("url_count", 0) >= 2 or signals.get("contact_spam"):
            score += 0.30
        elif signals.get("url_count", 0) == 1 or signals.get("phone_count", 0):
            score += 0.15

        # Amplifiers
        if signals.get("leet_detected") and toxic_count:
            score += 0.10
        if signals.get("caps_ratio", 0) > 0.7 and len(matched):
            score += 0.05
        if signals.get("repetition_abuse") and toxic_count:
            score += 0.05

        return min(score, 1.0)
