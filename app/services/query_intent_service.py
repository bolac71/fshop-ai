from dataclasses import dataclass, field
import re
import unicodedata


@dataclass
class ProductQueryIntent:
    query_terms: list[str] = field(default_factory=list)
    matched_categories: list[str] = field(default_factory=list)
    concept_terms: list[str] = field(default_factory=list)
    strict_category_filter: bool = False


class QueryIntentService:
    STOPWORDS = {
        "co", "khong", "khong", "nhi", "khong", "la", "cho", "toi", "minh", "giup",
        "cua", "hang", "shop", "fshop", "xin", "chao", "voi", "ve", "di", "nao",
        "please", "help", "need", "want", "co", "ban", "khong", "nhe", "a", "ah",
    }

    CATEGORY_QUALIFIERS = {
        "nam", "nu", "tre", "em", "kids", "kid", "women", "woman", "men", "man",
    }

    def _normalize_text(self, text: str) -> str:
        text = (text or "").lower()
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _tokenize(self, text: str) -> list[str]:
        normalized = self._normalize_text(text)
        return [t for t in normalized.split() if len(t) >= 2 and t not in self.STOPWORDS]

    def _category_score(self, query_terms: list[str], category_name: str) -> float:
        if not category_name:
            return 0.0
        category_terms = self._tokenize(category_name)
        if not category_terms or not query_terms:
            return 0.0

        overlap = len(set(query_terms) & set(category_terms))
        overlap_ratio = overlap / max(len(set(category_terms)), 1)

        q_text = " ".join(query_terms)
        c_text = " ".join(category_terms)
        phrase_bonus = 1.0 if c_text and c_text in q_text else 0.0
        return overlap_ratio + phrase_bonus

    def _extract_concept_terms(self, categories: list[str]) -> list[str]:
        concept = set()
        for category in categories:
            for token in self._tokenize(category):
                if token not in self.CATEGORY_QUALIFIERS:
                    concept.add(token)
        return sorted(concept)

    def analyze_product_query(self, query: str, candidate_metas: list[dict]) -> ProductQueryIntent:
        query_terms = self._tokenize(query)
        if not query_terms:
            return ProductQueryIntent()

        categories = sorted(
            {
                str(meta.get("category_name") or meta.get("category") or "").strip()
                for meta in candidate_metas
                if str(meta.get("category_name") or meta.get("category") or "").strip()
            }
        )
        if not categories:
            return ProductQueryIntent(query_terms=query_terms)

        scored = []
        for category in categories:
            score = self._category_score(query_terms, category)
            if score > 0:
                scored.append((category, score))

        if not scored:
            return ProductQueryIntent(query_terms=query_terms)

        scored.sort(key=lambda item: item[1], reverse=True)
        top_score = scored[0][1]
        threshold = max(0.75, top_score * 0.8)
        matched_categories = [name for name, score in scored if score >= threshold]

        if not matched_categories:
            return ProductQueryIntent(query_terms=query_terms)

        return ProductQueryIntent(
            query_terms=query_terms,
            matched_categories=matched_categories,
            concept_terms=self._extract_concept_terms(matched_categories),
            strict_category_filter=True,
        )

    def category_or_name_match(self, meta: dict, intent: ProductQueryIntent) -> bool:
        if not intent.strict_category_filter:
            return True

        category = self._normalize_text(str(meta.get("category_name") or meta.get("category") or ""))
        name = self._normalize_text(str(meta.get("name") or ""))

        matched_category_norm = [self._normalize_text(c) for c in intent.matched_categories]
        if any(category == c or c in category for c in matched_category_norm):
            return True

        if intent.concept_terms:
            return all(term in name or term in category for term in intent.concept_terms)

        return False
