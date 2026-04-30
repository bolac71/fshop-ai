import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


def normalize_vietnamese_text(text: str) -> str:
    text = (text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class VoiceQueryIntent:
    original_query: str
    normalized_query: str
    rewritten_query: str
    intent: str = "product_search"
    filters: dict[str, Any] = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list)
    visual_query: bool = False


class VoiceQueryUnderstandingService:
    STOPWORDS = {
        "toi", "minh", "em", "anh", "chi", "ban", "shop", "fshop", "muon",
        "can", "tim", "kiem", "mua", "cho", "co", "khong", "giup", "voi",
        "san", "pham", "hang", "cai", "mot", "may", "nhe", "a", "ah",
    }

    SYNONYM_PHRASES = {
        "ao phong": "ao thun",
        "t shirt": "ao thun",
        "tee": "ao thun",
        "tee shirt": "ao thun",
        "quan jean": "quan jeans",
        "jean": "jeans",
        "denim": "jeans",
        "sneaker": "giay the thao",
        "sneakers": "giay the thao",
        "tui xach": "tui",
        "tui deo cheo": "tui deo cheo",
        "non": "mu",
        "be": "mau be",
        "kem": "mau kem",
        "trang nga": "mau trang",
        "xanh navy": "xanh dam",
    }

    CATEGORY_TERMS = {
        "ao thun": ["ao phong", "t shirt", "tee shirt"],
        "ao so mi": ["so mi"],
        "ao khoac": ["jacket"],
        "quan jeans": ["quan jean", "denim"],
        "quan short": ["short"],
        "vay": ["dam"],
        "giay": ["sneaker", "sneakers"],
        "balo": ["cap sach"],
        "mu": ["non"],
        "tui deo cheo": ["mini bag"],
        "tui": ["tui xach"],
    }

    COLOR_TERMS = {
        "den": ["den", "black"],
        "trang": ["trang", "white"],
        "xanh": ["xanh", "blue"],
        "xanh dam": ["xanh dam", "navy"],
        "do": ["do", "red"],
        "hong": ["hong", "pink"],
        "vang": ["vang", "yellow"],
        "nau": ["nau", "brown"],
        "be": ["be", "kem", "beige"],
        "xam": ["xam", "ghi", "gray", "grey"],
    }

    SIZE_TERMS = {"xs", "s", "m", "l", "xl", "xxl", "free size", "freesize"}
    GENDER_TERMS = {
        "nam": ["nam", "men", "male"],
        "nu": ["nu", "women", "female"],
        "unisex": ["unisex"],
        "tre em": ["tre em", "be trai", "be gai", "kids", "kid"],
    }
    VISUAL_HINTS = {"giong", "kieu", "form", "oversize", "phong", "cach", "streetwear", "vintage"}

    def analyze(self, query: str) -> VoiceQueryIntent:
        normalized = normalize_vietnamese_text(query)
        rewritten = self._rewrite_synonyms(normalized)
        tokens = [t for t in rewritten.split() if len(t) >= 2 and t not in self.STOPWORDS]

        filters: dict[str, Any] = {}
        category = self._match_phrase_map(rewritten, self.CATEGORY_TERMS)
        color = self._match_phrase_map(rewritten, self.COLOR_TERMS)
        gender = self._match_phrase_map(rewritten, self.GENDER_TERMS)
        size = self._match_size(rewritten)
        price = self._extract_price(rewritten)

        if category:
            filters["category"] = category
        if color:
            filters["color"] = color
        if gender:
            filters["gender"] = gender
        if size:
            filters["size"] = size
        if price:
            filters["price"] = price

        visual_query = any(hint in tokens for hint in self.VISUAL_HINTS)
        return VoiceQueryIntent(
            original_query=query,
            normalized_query=normalized,
            rewritten_query=rewritten,
            filters=filters,
            tokens=tokens,
            visual_query=visual_query,
        )

    def _rewrite_synonyms(self, normalized: str) -> str:
        rewritten = f" {normalized} "
        for source, target in sorted(self.SYNONYM_PHRASES.items(), key=lambda item: len(item[0]), reverse=True):
            rewritten = rewritten.replace(f" {source} ", f" {target} ")
        return re.sub(r"\s+", " ", rewritten).strip()

    def _match_phrase_map(self, text: str, phrase_map: dict[str, list[str]]) -> str | None:
        padded = f" {text} "
        for canonical, aliases in sorted(phrase_map.items(), key=lambda item: len(item[0]), reverse=True):
            if f" {canonical} " in padded:
                return canonical
            if any(f" {alias} " in padded for alias in aliases):
                return canonical
        return None

    def _match_size(self, text: str) -> str | None:
        padded = f" {text} "
        for size in sorted(self.SIZE_TERMS, key=len, reverse=True):
            if f" {size} " in padded:
                return size.upper() if len(size) <= 3 else size
        return None

    def _extract_price(self, text: str) -> dict[str, float] | None:
        match = re.search(r"(duoi|nho hon|toi da)\s+(\d+(?:[.,]\d+)?)\s*(trieu|k|nghin|ngan)?", text)
        if match:
            return {"max": self._price_to_vnd(match.group(2), match.group(3))}

        match = re.search(r"(tren|hon|tu)\s+(\d+(?:[.,]\d+)?)\s*(trieu|k|nghin|ngan)?", text)
        if match:
            return {"min": self._price_to_vnd(match.group(2), match.group(3))}

        return None

    def _price_to_vnd(self, value: str, unit: str | None) -> float:
        amount = float(value.replace(",", "."))
        if unit == "trieu":
            return amount * 1_000_000
        if unit in {"k", "nghin", "ngan"}:
            return amount * 1_000
        return amount
