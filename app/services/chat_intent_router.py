import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field


ChatIntent = Literal[
    "small_talk",
    "product_discovery",
    "product_qa",
    "product_compare",
    "product_search",
    "size_advice",
    "color_question",
    "policy",
    "order",
    "unknown",
]

RouteSource = Literal["guard", "semantic", "llm", "fallback"]


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: ChatIntent
    confidence: float = 0.0
    search_query: str = ""
    requires_context: bool = False
    follow_up: bool = False
    entities: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    source: RouteSource = "fallback"


class ChatIntentRouter:
    PRODUCT_ANCHOR_HINTS = {
        "jean", "jeans", "quan", "ao", "vay", "dam", "giay", "dep", "hoodie",
        "shirt", "balo", "tee", "thun", "polo", "jacket", "sweater", "short",
        "skirt", "dirtycoins", "converse", "nike", "adidas", "mlb", "new",
        "balance", "core", "vans", "tui", "mu", "non",
    }

    PRODUCT_PHRASES = {
        "san pham", "mua", "tim", "tim kiem", "goi y", "de xuat",
        "phoi do", "mix match", "so sanh", "gia", "bao nhieu tien", "con hang",
        "ton kho", "recommend", "suggest", "cho minh xem", "outfit",
    }

    DISCOVERY_CHANGE_TERMS = {
        "khac", "cai khac", "mau khac", "san pham khac", "goi y lai",
        "de xuat lai", "phu hop cho toi", "phu hop voi toi", "cho toi",
    }

    PRODUCT_QA_TERMS = {
        "hop", "phu hop", "mac", "di hoc", "di choi", "di lam", "di tiec",
        "phoi", "mix", "style", "vibe", "nu", "nam", "unisex", "oversize",
        "rong", "om", "form", "da ngam", "ton da", "dep khong", "on khong",
        "co hop khong", "nen mac", "mua duoc khong",
    }

    REFERENCE_TERMS = {
        "nay", "kia", "do", "cai nay", "mau nay", "san pham nay", "ao nay",
        "quan nay", "vay nay", "dam nay", "giay nay", "mu nay", "non nay",
        "cai dau", "cai thu 1", "cai thu nhat", "cai thu 2", "cai thu hai",
        "cai thu 3", "cai thu ba", "mau dau", "mau thu 2", "mau thu hai",
    }

    def __init__(
        self,
        embeddings,
        llm_fast=None,
        routes_path: str | Path | None = None,
        semantic_threshold: float | None = None,
        llm_threshold: float | None = None,
        debug: bool | None = None,
    ):
        self.embeddings = embeddings
        self.llm_fast = llm_fast
        self.semantic_threshold = semantic_threshold or float(os.getenv("AI_INTENT_SEMANTIC_THRESHOLD", "0.72"))
        self.llm_threshold = llm_threshold or float(os.getenv("AI_INTENT_LLM_THRESHOLD", "0.55"))
        self.debug = (
            debug
            if debug is not None
            else os.getenv("AI_INTENT_ROUTER_DEBUG", "true").strip().lower() in {"1", "true", "yes", "on"}
        )
        self.routes_path = Path(routes_path) if routes_path else Path(__file__).resolve().parents[1] / "data" / "chat_intent_routes.json"
        self.routes = self._load_routes()
        self.route_vectors = self._embed_routes()
        self.llm_prompt = ChatPromptTemplate.from_template(
            """
            Bạn là bộ định tuyến intent cho chatbot thời trang FShop.
            Hãy trả về JSON hợp lệ duy nhất, không markdown, không giải thích.

            Capability hợp lệ:
            - small_talk: chào hỏi, cảm ơn, tạm biệt, hỏi bot là ai hoặc bot giúp được gì.
            - product_discovery: tìm/gợi ý sản phẩm mới, hỏi giá/tồn kho/sản phẩm mới.
            - product_qa: hỏi tiếp/tư vấn về sản phẩm đang nói tới: hợp không, nam/nữ mặc được không, đi học/đi chơi được không, phối với gì, size/màu/form.
            - product_compare: so sánh hai hoặc nhiều sản phẩm.
            - policy: hỏi chính sách shop, ship, đổi trả, thanh toán, liên hệ, cửa hàng.
            - order: hỏi đơn hàng, vận đơn, lịch sử mua, trạng thái/hủy đơn.
            - unknown: câu rỗng, rác, mơ hồ, không đủ nhu cầu.

            User question: {question}
            Recent history:
            {history}
            Session state:
            {session_state}

            Schema:
            {{
              "intent": "small_talk" | "product_discovery" | "product_qa" | "product_compare" | "policy" | "order" | "unknown",
              "confidence": number,
              "search_query": string,
              "requires_context": boolean,
              "follow_up": boolean,
              "entities": object,
              "reason": string
            }}
            """
        )

    def route(self, question: str, history: list | None = None, session_state: Any = None) -> RouteDecision:
        normalized = self.normalize_text(question)
        if not normalized:
            return self._decision("unknown", question, 1.0, "empty input", "guard")

        guard = self._guard_route(question, normalized)
        if guard:
            self._log(question, guard, [])
            return guard

        semantic_decision, top_candidates = self._semantic_route(question)
        if semantic_decision and semantic_decision.confidence >= self.semantic_threshold:
            final = self._post_guard(semantic_decision, question)
            self._log(question, final, top_candidates)
            return final

        llm_decision = self._llm_route(question, history or [], session_state)
        if llm_decision and llm_decision.confidence >= self.llm_threshold:
            final = self._post_guard(llm_decision, question)
            self._log(question, final, top_candidates)
            return final

        fallback = self._decision("unknown", question, 0.0, "no confident route", "fallback")
        final = self._post_guard(fallback, question)
        self._log(question, final, top_candidates)
        return final

    def has_product_signal(self, question: str) -> bool:
        q = self.normalize_text(question)
        if not q:
            return False
        tokens = set(q.split())
        if any(token in self.PRODUCT_ANCHOR_HINTS for token in tokens):
            return True
        return any(phrase in q for phrase in self.PRODUCT_PHRASES)

    def normalize_text(self, text: str) -> str:
        text = (text or "").lower()
        text = text.replace("đ", "d")
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _load_routes(self) -> dict[str, list[str]]:
        try:
            with self.routes_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception as exc:
            print(f"Intent router route file failed to load: {exc}")
            data = {}

        routes: dict[str, list[str]] = {}
        for intent in ("small_talk", "product_discovery", "product_qa", "product_compare", "policy", "order", "unknown"):
            examples = data.get(intent, [])
            routes[intent] = [str(item).strip() for item in examples if str(item).strip()]
        return routes

    def _embed_routes(self) -> dict[str, np.ndarray]:
        vectors: dict[str, np.ndarray] = {}
        for intent, examples in self.routes.items():
            if not examples:
                continue
            normalized_examples = [self.normalize_text(example) or example for example in examples]
            embedded = np.asarray(self.embeddings.embed_documents(normalized_examples), dtype=np.float32)
            vectors[intent] = self._normalize_vectors(embedded)
        return vectors

    def _semantic_route(self, question: str) -> tuple[RouteDecision | None, list[dict[str, Any]]]:
        if not self.route_vectors:
            return None, []

        query = self.normalize_text(question) or question
        query_vector = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
        query_vector = self._normalize_vectors(query_vector.reshape(1, -1))[0]

        candidates = []
        for intent, vectors in self.route_vectors.items():
            similarities = vectors @ query_vector
            score = float(np.max(similarities)) if similarities.size else 0.0
            candidates.append({"intent": intent, "score": score})

        candidates.sort(key=lambda item: item["score"], reverse=True)
        if not candidates:
            return None, []

        best = candidates[0]
        decision = self._decision(
            intent=best["intent"],
            question=question,
            confidence=best["score"],
            reason="semantic example route",
            source="semantic",
        )
        return decision, candidates[:3]

    def _llm_route(self, question: str, history: list, session_state: Any) -> RouteDecision | None:
        if not self.llm_fast:
            return None

        history_text = "\n".join(
            f"{getattr(item, 'role', item.get('role', ''))}: {getattr(item, 'content', item.get('content', ''))}"
            if isinstance(item, dict)
            else f"{getattr(item, 'role', '')}: {getattr(item, 'content', '')}"
            for item in history[-8:]
        )
        try:
            session_text = session_state.model_dump() if hasattr(session_state, "model_dump") else session_state
            raw = (self.llm_prompt | self.llm_fast | StrOutputParser()).invoke(
                {
                    "question": question,
                    "history": history_text,
                    "session_state": json.dumps(session_text or {}, ensure_ascii=False),
                }
            )
            payload = json.loads(self._extract_json_payload(raw))
            payload["intent"] = self._normalize_intent(str(payload.get("intent") or "unknown"))
            payload["confidence"] = float(payload.get("confidence", 0.0) or 0.0)
            payload["search_query"] = str(payload.get("search_query") or question).strip()
            payload["requires_context"] = bool(payload.get("requires_context", False))
            payload["follow_up"] = bool(payload.get("follow_up", False))
            payload["entities"] = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
            payload["reason"] = str(payload.get("reason") or "llm route")
            payload["source"] = "llm"
            return RouteDecision.model_validate(payload)
        except Exception as exc:
            if self.debug:
                print(f"CHAT INTENT ROUTER | llm route failed: {exc}")
            return None

    def _guard_route(self, question: str, normalized: str) -> RouteDecision | None:
        if self._is_small_talk_guard(normalized):
            return self._decision("small_talk", question, 0.96, "explicit small talk", "guard")

        if re.search(r"(?:order|don hang|don|ma don|#)\s*\d+", normalized):
            return self._decision("order", question, 0.98, "explicit order id", "guard")

        order_terms = [
            "don hang", "don cua toi", "don cua minh", "don minh", "kiem tra don",
            "theo doi don", "trang thai don", "lich su mua", "van don", "huy don",
            "track my order", "package status",
        ]
        if any(term in normalized for term in order_terms):
            return self._decision("order", question, 0.93, "explicit order term", "guard")

        policy_terms = [
            "chinh sach", "phi ship", "van chuyen", "doi tra", "hoan tien",
            "thanh toan", "cod", "dia chi", "cua hang", "gio mo cua", "lien he",
        ]
        if any(term in normalized for term in policy_terms):
            return self._decision("policy", question, 0.92, "explicit policy term", "guard")

        if self._is_compare_question(normalized):
            return self._decision("product_compare", question, 0.90, "explicit comparison", "guard", requires_context=True)

        if self._is_new_recommendation_request(normalized):
            return self._decision("product_discovery", question, 0.92, "new recommendation or preference correction", "guard")

        if self._is_product_followup(normalized):
            return self._decision("product_qa", question, 0.86, "contextual product follow-up", "guard", requires_context=True, follow_up=True)

        if self.has_product_signal(question):
            return self._decision("product_discovery", question, 0.94, "explicit product discovery signal", "guard")

        return None

    def _is_small_talk_guard(self, normalized: str) -> bool:
        if not normalized:
            return False

        recipient = r"(?:ban|shop|finn|ad|admin|bot|nha|nhe|oi|a|em|anh|chi)"
        greeting = rf"^(?:xin\s+)?(?:chao|hello|hi|hey|alo)(?:\s+{recipient})*$"
        thanks = rf"^(?:cam on|thanks|thank you|tks|thank)(?:\s+{recipient})*$"
        goodbye = rf"^(?:tam biet|bye|goodbye|hen gap lai)(?:\s+{recipient})*$"
        identity_or_help = (
            r"^(?:ban|finn|bot|chatbot)(?:\s+nay)?\s+"
            r"(?:la ai|la gi|giup duoc gi|lam duoc gi|co the ho tro gi|ho tro gi)$"
        )
        social = rf"^(?:{recipient}\s+)?(?:khoe khong|hom nay the nao)$"

        return any(
            re.match(pattern, normalized)
            for pattern in (greeting, thanks, goodbye, identity_or_help, social)
        )

    def _is_compare_question(self, normalized: str) -> bool:
        if "so sanh" in normalized:
            return True
        return bool(re.search(r"\b(cai|mau|san pham)?\s*(nao|gi)\s+(dep|tot|hop|dang mua|nen mua)\s+hon\b", normalized))

    def _is_new_recommendation_request(self, normalized: str) -> bool:
        if not normalized:
            return False
        has_discovery_phrase = any(self._contains_term(normalized, term) for term in self.PRODUCT_PHRASES)
        has_change_term = any(self._contains_term(normalized, term) for term in self.DISCOVERY_CHANGE_TERMS)
        has_gender_preference = bool(re.search(r"\b(toi|minh|em|anh|chi)\s+(la\s+)?(nam|nu)\b", normalized))
        if has_discovery_phrase and (has_change_term or has_gender_preference):
            return True
        if has_gender_preference and has_change_term:
            return True
        return False

    def _is_product_followup(self, normalized: str) -> bool:
        if not normalized:
            return False
        if re.search(r"\b(phoi|mix)\s+(voi|cung)\b", normalized):
            return True
        has_reference = any(self._contains_term(normalized, term) for term in self.REFERENCE_TERMS)
        has_qa_term = any(self._contains_term(normalized, term) for term in self.PRODUCT_QA_TERMS)
        has_size_or_color = bool(re.search(r"\b(cao|nang|kg|cm|1m\d{2}|size|kich co|form|mau|color|phoi mau)\b", normalized))
        if has_reference and (has_qa_term or has_size_or_color):
            return True
        if has_qa_term and not self.has_product_signal(normalized):
            return True
        return has_size_or_color and not self.has_product_signal(normalized)

    def _contains_term(self, normalized: str, term: str) -> bool:
        normalized_term = self.normalize_text(term)
        if not normalized_term:
            return False
        return bool(re.search(rf"(^|\s){re.escape(normalized_term)}(\s|$)", normalized))

    def _post_guard(self, decision: RouteDecision, question: str) -> RouteDecision:
        if decision.intent in {"product_search", "product_discovery"} and not self.has_product_signal(question) and not decision.follow_up and not decision.requires_context:
            return self._decision(
                "unknown",
                question,
                min(decision.confidence, 0.60),
                f"downgraded from product discovery: {decision.reason}",
                "fallback",
            )
        return decision

    def _decision(
        self,
        intent: ChatIntent,
        question: str,
        confidence: float,
        reason: str,
        source: RouteSource,
        requires_context: bool = False,
        follow_up: bool = False,
        entities: dict[str, Any] | None = None,
    ) -> RouteDecision:
        return RouteDecision(
            intent=intent,
            confidence=max(0.0, min(float(confidence), 1.0)),
            search_query=question,
            requires_context=requires_context,
            follow_up=follow_up,
            entities=entities or {},
            reason=reason,
            source=source,
        )

    def _normalize_intent(self, intent: str) -> ChatIntent:
        normalized = self.normalize_text(intent).replace(" ", "_")
        mapping = {
            "product": "product_search",
            "product_search": "product_search",
            "product_discovery": "product_discovery",
            "discovery": "product_discovery",
            "product_qa": "product_qa",
            "product_question": "product_qa",
            "product_advice": "product_qa",
            "style_advice": "product_qa",
            "size": "product_qa",
            "size_advice": "product_qa",
            "color": "product_qa",
            "color_question": "product_qa",
            "product_compare": "product_compare",
            "compare": "product_compare",
            "comparison": "product_compare",
            "policy": "policy",
            "order": "order",
            "small_talk": "small_talk",
            "greeting": "small_talk",
            "unknown": "unknown",
        }
        return mapping.get(normalized, "unknown")

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def _extract_json_payload(self, text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return "{}"
        code_fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
        if code_fence_match:
            raw = code_fence_match.group(1).strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return raw
        return raw[start:end + 1]

    def _log(self, question: str, decision: RouteDecision, top_candidates: list[dict[str, Any]]) -> None:
        if not self.debug:
            return
        compact_candidates = [
            {"intent": item["intent"], "score": round(float(item["score"]), 4)}
            for item in top_candidates
        ]
        print(
            "CHAT INTENT ROUTER | "
            f"question={question!r} intent={decision.intent} confidence={decision.confidence:.3f} "
            f"source={decision.source} reason={decision.reason!r} top={compact_candidates}"
        )
