from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field

from app.models.schemas import ChatSessionState, ProductInfo
from app.services.chat_planner import ChatPlan, ReferenceSpec, SearchConstraints


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    products: list[ProductInfo] = Field(default_factory=list)
    docs: list[Any] = Field(default_factory=list)
    context: str = ""
    needs_clarification: bool = False
    clarification_question: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatToolExecutor:
    def __init__(self, rag_service):
        self.rag = rag_service

    def resolve_reference(self, reference: ReferenceSpec, state: ChatSessionState | dict | None) -> list[int]:
        current_state = state if isinstance(state, ChatSessionState) else ChatSessionState.model_validate(state or {})
        if reference.product_id:
            return [reference.product_id]
        if reference.type == "active_product" and current_state.active_product_id:
            return [current_state.active_product_id]
        if reference.type == "list_item":
            index = reference.index if reference.index is not None else 0
            if 0 <= index < len(current_state.last_product_ids):
                return [current_state.last_product_ids[index]]
            return []
        if reference.type == "product_list":
            return list(current_state.last_product_ids or [])
        return []

    def get_products_by_ids(self, product_ids: list[int]) -> ToolResult:
        docs = self.rag._get_product_docs_by_ids(product_ids)
        products = self.rag._build_products_from_docs(docs)
        return ToolResult(
            kind="products_by_ids",
            products=products,
            docs=docs,
            context=self.rag._build_product_context_text(docs),
            metadata={"product_ids": [product.id for product in products]},
        )

    def search_products(self, constraints: SearchConstraints, limit: int | None = None) -> ToolResult:
        query = self.build_search_query(constraints)
        docs, filters = self.rag._search_product_docs(
            query=query,
            constraints=constraints.model_dump(),
            exclude_product_ids=constraints.exclude_product_ids,
            limit=limit,
        )
        products = self.rag._build_products_from_docs(docs)
        return ToolResult(
            kind="product_search",
            products=products,
            docs=docs,
            context=self.rag._build_product_context_text(docs),
            metadata={
                "query": query,
                "filters": filters,
                "exclude_product_ids": constraints.exclude_product_ids,
            },
        )

    def resolve_product_advice(self, plan: ChatPlan, state: ChatSessionState | dict | None) -> ToolResult:
        product_ids = self.resolve_reference(plan.reference, state)
        if not product_ids:
            return ToolResult(
                kind="product_advice",
                needs_clarification=True,
                clarification_question="Bạn đang hỏi sản phẩm nào vậy? Bạn có thể chọn một sản phẩm trong danh sách vừa xem hoặc nhắn tên sản phẩm cho Finn nhé.",
            )
        return self.get_products_by_ids(product_ids[:1])

    def resolve_compare(self, plan: ChatPlan, state: ChatSessionState | dict | None) -> ToolResult:
        product_ids = self.resolve_reference(plan.reference, state)
        if len(product_ids) < 2:
            current_state = state if isinstance(state, ChatSessionState) else ChatSessionState.model_validate(state or {})
            product_ids = list(current_state.last_product_ids or [])[:2]
        if len(product_ids) < 2:
            return ToolResult(
                kind="compare_products",
                needs_clarification=True,
                clarification_question="Bạn muốn so sánh những sản phẩm nào? Bạn có thể nói: so sánh mẫu đầu và mẫu thứ 2 nhé.",
            )
        return self.get_products_by_ids(product_ids[:3])

    def build_search_query(self, constraints: SearchConstraints) -> str:
        parts = [
            constraints.query,
            self._gender_text(constraints.gender),
            constraints.category,
            constraints.brand,
            constraints.color,
            constraints.size,
            constraints.style,
            constraints.occasion,
        ]
        return " ".join(str(part).strip() for part in parts if part).strip() or "sản phẩm thời trang"

    def _gender_text(self, gender: str | None) -> str:
        if gender == "male":
            return "nam"
        if gender == "female":
            return "nữ"
        if gender == "unisex":
            return "unisex"
        return ""
