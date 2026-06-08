import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.schemas import ChatSessionState, ParsedChatQuery


ResolvedSubject = Literal["active_product", "product_list_item", "new_search", "none"]


class ContextResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: ResolvedSubject = "none"
    product_id: int | None = None
    product_ids: list[int] = Field(default_factory=list)
    needs_retrieval: bool = False
    needs_clarification: bool = False
    reason: str = ""


class ChatContextResolver:
    """Resolve conversational references such as "cái này" or "cái thứ 2"."""

    def resolve(
        self,
        question: str,
        parsed_query: ParsedChatQuery,
        session_state: ChatSessionState | dict | None,
    ) -> ContextResolution:
        state = session_state if isinstance(session_state, ChatSessionState) else ChatSessionState.model_validate(session_state or {})
        q = self.normalize_text(question)
        last_product_ids = [int(pid) for pid in (state.last_product_ids or []) if pid]

        if parsed_query.intent == "product_discovery":
            return ContextResolution(subject="new_search", needs_retrieval=True, reason="new product discovery")

        if parsed_query.intent == "product_compare":
            product_ids = self._resolve_compare_ids(q, state)
            return ContextResolution(
                subject="product_list_item" if product_ids else "none",
                product_ids=product_ids,
                needs_retrieval=bool(product_ids),
                needs_clarification=len(product_ids) < 2,
                reason="comparison references resolved" if len(product_ids) >= 2 else "comparison needs at least two products",
            )

        if parsed_query.intent != "product_qa":
            return ContextResolution(reason="capability does not need product context")

        ordinal_index = self._ordinal_index(q)
        if ordinal_index is not None:
            if 0 <= ordinal_index < len(last_product_ids):
                return ContextResolution(
                    subject="product_list_item",
                    product_id=last_product_ids[ordinal_index],
                    product_ids=[last_product_ids[ordinal_index]],
                    needs_retrieval=True,
                    reason=f"resolved ordinal product reference index={ordinal_index}",
                )
            return ContextResolution(
                subject="none",
                needs_clarification=True,
                reason="ordinal product reference not available in session",
            )

        if state.active_product_id:
            return ContextResolution(
                subject="active_product",
                product_id=state.active_product_id,
                product_ids=[state.active_product_id],
                needs_retrieval=True,
                reason="using active product context",
            )

        if last_product_ids:
            return ContextResolution(
                subject="product_list_item",
                product_id=last_product_ids[0],
                product_ids=[last_product_ids[0]],
                needs_retrieval=True,
                reason="using first product from previous result list",
            )

        return ContextResolution(
            subject="none",
            needs_clarification=True,
            reason="product follow-up has no active product context",
        )

    def normalize_text(self, text: str) -> str:
        text = (text or "").lower()
        text = text.replace("đ", "d")
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _ordinal_index(self, normalized: str) -> int | None:
        patterns = [
            (0, [r"\b(cai|mau|san pham)?\s*(dau|thu nhat|thu 1|so 1)\b"]),
            (1, [r"\b(cai|mau|san pham)?\s*(thu hai|thu 2|so 2)\b"]),
            (2, [r"\b(cai|mau|san pham)?\s*(thu ba|thu 3|so 3)\b"]),
        ]
        for index, index_patterns in patterns:
            if any(re.search(pattern, normalized) for pattern in index_patterns):
                return index
        return None

    def _resolve_compare_ids(self, normalized: str, state: ChatSessionState) -> list[int]:
        last_product_ids = [int(pid) for pid in (state.last_product_ids or []) if pid]
        if len(last_product_ids) >= 2:
            indices = []
            for pattern_index in (0, 1, 2):
                if self._mentions_ordinal(normalized, pattern_index):
                    indices.append(pattern_index)
            if len(indices) >= 2:
                return [last_product_ids[index] for index in indices if index < len(last_product_ids)]
            return last_product_ids[:2]
        if state.active_product_id:
            return [state.active_product_id]
        return []

    def _mentions_ordinal(self, normalized: str, index: int) -> bool:
        old = self._ordinal_index(normalized)
        if old == index:
            return True
        terms = {
            0: ["dau", "thu nhat", "thu 1", "so 1"],
            1: ["thu hai", "thu 2", "so 2"],
            2: ["thu ba", "thu 3", "so 3"],
        }
        return any(term in normalized for term in terms.get(index, []))
