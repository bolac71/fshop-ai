import json
import re
import unicodedata
from typing import Any, Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field

from app.models.schemas import ChatSessionState


ChatTask = Literal[
    "recommend_products",
    "product_advice",
    "compare_products",
    "policy_qa",
    "order_qa",
    "small_talk",
    "clarify",
]

ReferenceType = Literal["none", "active_product", "list_item", "product_list", "explicit_product"]


class ReferenceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ReferenceType = "none"
    index: int | None = None
    product_id: int | None = None
    text: str | None = None


class SearchConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = ""
    category: str | None = None
    brand: str | None = None
    gender: Literal["male", "female", "unisex"] | None = None
    color: str | None = None
    size: str | None = None
    style: str | None = None
    occasion: str | None = None
    exclude_product_ids: list[int] = Field(default_factory=list)


class UserPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Literal["male", "female", "unisex"] | None = None
    style: str | None = None
    occasion: str | None = None
    fit: str | None = None
    colors: list[str] = Field(default_factory=list)
    negative_preferences: list[str] = Field(default_factory=list)


class ChatPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: ChatTask
    is_new_task: bool = False
    uses_previous_context: bool = False
    reference: ReferenceSpec = Field(default_factory=ReferenceSpec)
    search: SearchConstraints = Field(default_factory=SearchConstraints)
    user_preferences: UserPreferences = Field(default_factory=UserPreferences)
    needs_clarification: bool = False
    clarification_question: str | None = None
    confidence: float = 0.0
    reason: str = ""
    source: str = "llm"


class ChatPlanner:
    def __init__(self, llm_fast=None, debug: bool = True):
        self.llm_fast = llm_fast
        self.debug = debug
        self.prompt = ChatPromptTemplate.from_template(
            """
            Bạn là conversation planner cho chatbot mua sắm thời trang Finn.
            Trả về DUY NHẤT một JSON hợp lệ, không markdown, không giải thích.

            Nhiệm vụ hợp lệ:
            - recommend_products: tìm/gợi ý sản phẩm hoặc outfit mới.
            - product_advice: hỏi tiếp/tư vấn về sản phẩm đang active hoặc item trong danh sách trước.
            - compare_products: so sánh nhiều sản phẩm.
            - policy_qa: hỏi chính sách shop.
            - order_qa: hỏi đơn hàng/trạng thái đơn.
            - small_talk: chào hỏi, cảm ơn, tạm biệt, hỏi bot là ai.
            - clarify: câu mơ hồ/rác/thiếu thông tin.

            Quy tắc quan trọng:
            - "tôi là nam/nữ, gợi ý cái khác/phù hợp" là recommend_products, is_new_task=true, uses_previous_context=false.
            - "nữ/nam mặc có hợp không", "mặc đi học được không", "phối với gì" là product_advice nếu có active product/list trước.
            - "cái đầu/cái thứ 2" dùng reference list_item với index 0/1.
            - "so sánh cái đầu và cái thứ 2" là compare_products.
            - "shop có bán quần jeans không" là recommend_products, category="jeans".
            - Nếu user yêu cầu "cái khác", đưa last_product_ids vào search.exclude_product_ids.

            User question: {question}
            Recent history:
            {history}
            Session state:
            {session_state}

            JSON schema:
            {{
              "task": "recommend_products" | "product_advice" | "compare_products" | "policy_qa" | "order_qa" | "small_talk" | "clarify",
              "is_new_task": boolean,
              "uses_previous_context": boolean,
              "reference": {{"type": "none" | "active_product" | "list_item" | "product_list" | "explicit_product", "index": number | null, "product_id": number | null, "text": string | null}},
              "search": {{"query": string, "category": string | null, "brand": string | null, "gender": "male" | "female" | "unisex" | null, "color": string | null, "size": string | null, "style": string | null, "occasion": string | null, "exclude_product_ids": [number]}},
              "user_preferences": {{"gender": "male" | "female" | "unisex" | null, "style": string | null, "occasion": string | null, "fit": string | null, "colors": [string], "negative_preferences": [string]}},
              "needs_clarification": boolean,
              "clarification_question": string | null,
              "confidence": number,
              "reason": string
            }}
            """
        )

    def plan(self, question: str, history: list | None, session_state: ChatSessionState | dict | None) -> ChatPlan:
        state = session_state if isinstance(session_state, ChatSessionState) else ChatSessionState.model_validate(session_state or {})
        guard = self._guard_plan(question, state)
        if guard:
            return guard

        if self.llm_fast:
            try:
                raw = (self.prompt | self.llm_fast | StrOutputParser()).invoke(
                    {
                        "question": question,
                        "history": self._format_history(history or []),
                        "session_state": json.dumps(state.model_dump(), ensure_ascii=False),
                    }
                )
                payload = json.loads(self._extract_json_payload(raw))
                payload["source"] = "llm"
                plan = ChatPlan.model_validate(payload)
                return self._post_process(plan, question, state)
            except Exception as exc:
                if self.debug:
                    print(f"CHAT PLANNER | llm plan failed: {exc}")

        return self._fallback_plan(question, state)

    def _guard_plan(self, question: str, state: ChatSessionState) -> ChatPlan | None:
        q = self.normalize_text(question)
        if not q:
            return ChatPlan(task="clarify", needs_clarification=True, confidence=1.0, reason="empty input", source="guard")
        if self._is_small_talk(q):
            return ChatPlan(task="small_talk", confidence=0.98, reason="small talk guard", source="guard")
        if re.search(r"(?:order|don hang|don|ma don|#)\s*\d+", q) or any(term in q for term in ["don hang", "kiem tra don", "trang thai don", "van don"]):
            return ChatPlan(task="order_qa", confidence=0.95, reason="order guard", source="guard")
        if any(term in q for term in ["chinh sach", "doi tra", "phi ship", "van chuyen", "thanh toan", "hoan tien", "lien he"]):
            return ChatPlan(task="policy_qa", confidence=0.93, reason="policy guard", source="guard")
        if self._is_compare(q):
            return ChatPlan(
                task="compare_products",
                uses_previous_context=True,
                reference=ReferenceSpec(type="product_list"),
                confidence=0.92,
                reason="comparison guard",
                source="guard",
            )
        if "san pham khach dinh kem de hoi" in q or (
            state.active_product_id and any(term in q for term in ["tu van", "tu van rieng", "san pham da chon", "dang xem"])
        ):
            return ChatPlan(
                task="product_advice",
                uses_previous_context=True,
                reference=ReferenceSpec(type="active_product", product_id=state.active_product_id),
                confidence=0.94,
                reason="attached product advice guard",
                source="guard",
            )
        if self._is_new_recommendation(q) or self._has_product_discovery_signal(q):
            search = SearchConstraints(query=question)
            prefs = UserPreferences()
            self._apply_gender(q, search, prefs)
            self._apply_category(q, search)
            self._apply_occasion(q, search, prefs)
            if self._asks_other(q):
                search.exclude_product_ids = state.last_product_ids or []
            return ChatPlan(
                task="recommend_products",
                is_new_task=True,
                uses_previous_context=False,
                search=search,
                user_preferences=prefs,
                confidence=0.92,
                reason="product discovery guard",
                source="guard",
            )
        return None

    def _fallback_plan(self, question: str, state: ChatSessionState) -> ChatPlan:
        q = self.normalize_text(question)
        search = SearchConstraints(query=question)
        prefs = UserPreferences()
        self._apply_gender(q, search, prefs)
        self._apply_category(q, search)
        self._apply_occasion(q, search, prefs)

        if self._is_compare(q):
            return ChatPlan(
                task="compare_products",
                uses_previous_context=True,
                reference=ReferenceSpec(type="product_list"),
                confidence=0.82,
                reason="fallback comparison",
                source="fallback",
            )

        if self._is_new_recommendation(q):
            if self._asks_other(q):
                search.exclude_product_ids = state.last_product_ids or []
            return ChatPlan(
                task="recommend_products",
                is_new_task=True,
                uses_previous_context=False,
                search=search,
                user_preferences=prefs,
                confidence=0.84,
                reason="fallback recommendation",
                source="fallback",
            )

        if self._is_product_advice(q):
            ref = ReferenceSpec(type="active_product")
            idx = self._ordinal_index(q)
            if idx is not None:
                ref = ReferenceSpec(type="list_item", index=idx)
            return ChatPlan(
                task="product_advice",
                uses_previous_context=True,
                reference=ref,
                search=search,
                user_preferences=prefs,
                confidence=0.80,
                reason="fallback product advice",
                source="fallback",
            )

        if self._has_product_discovery_signal(q):
            return ChatPlan(
                task="recommend_products",
                is_new_task=True,
                uses_previous_context=False,
                search=search,
                user_preferences=prefs,
                confidence=0.78,
                reason="fallback product discovery",
                source="fallback",
            )

        return ChatPlan(
            task="clarify",
            needs_clarification=True,
            clarification_question="Bạn muốn Finn tìm sản phẩm, tư vấn sản phẩm đang xem, so sánh hay hỏi chính sách/đơn hàng vậy?",
            confidence=0.45,
            reason="fallback unclear",
            source="fallback",
        )

    def _post_process(self, plan: ChatPlan, question: str, state: ChatSessionState) -> ChatPlan:
        q = self.normalize_text(question)
        if plan.task == "small_talk" and not self._is_small_talk(q):
            fallback = self._fallback_plan(question, state)
            if fallback.task != "clarify":
                fallback.reason = f"overrode non-small-talk llm plan: {plan.reason}"
                fallback.source = "fallback"
                return fallback
        if plan.task == "recommend_products":
            plan.is_new_task = True
            plan.uses_previous_context = False
            if not plan.search.query:
                plan.search.query = question
            self._apply_gender(q, plan.search, plan.user_preferences)
            self._apply_category(q, plan.search)
            self._apply_occasion(q, plan.search, plan.user_preferences)
            if self._asks_other(q) and not plan.search.exclude_product_ids:
                plan.search.exclude_product_ids = state.last_product_ids or []
        elif plan.task == "product_advice":
            if plan.reference.type == "none":
                idx = self._ordinal_index(q)
                plan.reference = ReferenceSpec(type="list_item", index=idx) if idx is not None else ReferenceSpec(type="active_product")
            plan.uses_previous_context = True
            self._apply_gender(q, plan.search, plan.user_preferences)
        elif plan.task == "compare_products":
            plan.uses_previous_context = True
            if plan.reference.type == "none":
                plan.reference = ReferenceSpec(type="product_list")
        return plan

    def _apply_gender(self, q: str, search: SearchConstraints, prefs: UserPreferences) -> None:
        if re.search(r"(^|\s)(toi|minh|em|anh|chi)?\s*(la\s+)?(nam|men|male)(\s|$)", q):
            search.gender = "male"
            prefs.gender = "male"
        elif re.search(r"(^|\s)(toi|minh|em|anh|chi)?\s*(la\s+)?(nu|women|female)(\s|$)", q):
            search.gender = "female"
            prefs.gender = "female"

    def _apply_category(self, q: str, search: SearchConstraints) -> None:
        category_aliases = [
            ("jeans", ["quan jeans", "jeans", "jean"]),
            ("ao thun", ["ao thun", "t shirt", "tee"]),
            ("quan", ["quan"]),
            ("vay", ["vay", "dam"]),
            ("giay", ["giay", "sneaker"]),
            ("mu", ["mu", "non"]),
            ("balo", ["balo"]),
            ("tui", ["tui"]),
        ]
        for category, aliases in category_aliases:
            if any(re.search(rf"(^|\s){re.escape(alias)}(\s|$)", q) for alias in aliases):
                search.category = category
                return

    def _apply_occasion(self, q: str, search: SearchConstraints, prefs: UserPreferences) -> None:
        for occasion, terms in {
            "đi học": ["di hoc", "school"],
            "đi làm": ["di lam", "cong so", "office"],
            "đi chơi": ["di choi", "date"],
            "đi tiệc": ["di tiec", "party"],
        }.items():
            if any(term in q for term in terms):
                search.occasion = occasion
                prefs.occasion = occasion
                return

    def _has_product_discovery_signal(self, q: str) -> bool:
        product_terms = ["ao", "quan", "vay", "dam", "giay", "dep", "mu", "non", "balo", "tui", "outfit", "san pham"]
        discovery_terms = ["tim", "goi y", "de xuat", "mua", "co ban", "cho xem", "recommend", "suggest", "khac"]
        return any(term in q for term in product_terms) or any(term in q for term in discovery_terms)

    def _is_new_recommendation(self, q: str) -> bool:
        return any(term in q for term in ["goi y", "de xuat", "recommend", "suggest", "outfit", "cho xem", "tim"]) or self._asks_other(q)

    def _is_product_advice(self, q: str) -> bool:
        return any(
            self._contains_term(q, term)
            for term in ["hop", "phu hop", "mac", "di hoc", "di lam", "di choi", "phoi", "mix", "size", "mau", "form"]
        )

    def _is_compare(self, q: str) -> bool:
        return "so sanh" in q or "cai nao" in q and any(term in q for term in ["hon", "tot", "dep", "hop"])

    def _asks_other(self, q: str) -> bool:
        return any(term in q for term in ["khac", "goi y lai", "de xuat lai", "doi sang"])

    def _contains_term(self, normalized: str, term: str) -> bool:
        normalized_term = self.normalize_text(term)
        if not normalized_term:
            return False
        return bool(re.search(rf"(^|\s){re.escape(normalized_term)}(\s|$)", normalized))

    def _ordinal_index(self, q: str) -> int | None:
        if re.search(r"\b(dau|thu nhat|thu 1|so 1)\b", q):
            return 0
        if re.search(r"\b(thu hai|thu 2|so 2)\b", q):
            return 1
        if re.search(r"\b(thu ba|thu 3|so 3)\b", q):
            return 2
        return None

    def _is_small_talk(self, q: str) -> bool:
        suffix = r"(ban|shop|finn|ad|admin|bot|nhe|nha|a|oi)"
        return bool(
            re.match(rf"^(xin\s+)?(chao|hello|hi|hey|alo)(\s+{suffix})*$", q)
            or re.match(rf"^(cam on|thanks|thank you|tam biet|bye)(\s+{suffix})*$", q)
            or re.search(r"\b(ban|finn|bot)\s+(la ai|giup duoc gi|lam duoc gi)\b", q)
        )

    def normalize_text(self, text: str) -> str:
        text = (text or "").lower()
        text = text.replace("đ", "d")
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _format_history(self, history: list, limit: int = 8) -> str:
        parts = []
        for item in history[-limit:]:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
            else:
                role = getattr(item, "role", "")
                content = getattr(item, "content", "")
            if content:
                parts.append(f"{role}: {content}")
        return "\n".join(parts)

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
