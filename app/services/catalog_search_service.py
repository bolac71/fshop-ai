import time
from dataclasses import asdict
from typing import Any

import torch
from importlib import import_module
from importlib.util import find_spec
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models as qd_models
from sentence_transformers import CrossEncoder

from app.core.config import (
    COLLECTION_PRODUCT_TEXT,
    RERANKER_MODEL,
    TEXT_EMBEDDING_MODEL,
)
from app.services.voice_query_utils import VoiceQueryIntent, VoiceQueryUnderstandingService, normalize_vietnamese_text

if find_spec("langchain_huggingface"):
    HuggingFaceEmbeddings = import_module("langchain_huggingface").HuggingFaceEmbeddings
else:
    HuggingFaceEmbeddings = import_module("langchain_community.embeddings").HuggingFaceEmbeddings


class CatalogSearchService:
    def __init__(self, client, rag_service=None, image_service=None):
        self.client = client
        self.image_service = image_service
        self.query_understanding = VoiceQueryUnderstandingService()

        if rag_service is not None:
            self.product_store = rag_service.product_store
            self.reranker = rag_service.reranker
            self.embeddings = getattr(rag_service, "embeddings", None)
            self.device = getattr(rag_service, "device", "cpu")
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embeddings = HuggingFaceEmbeddings(
                model_name=TEXT_EMBEDDING_MODEL,
                model_kwargs={"device": self.device},
            )
            self.product_store = QdrantVectorStore(
                client=self.client,
                collection_name=COLLECTION_PRODUCT_TEXT,
                embedding=self.embeddings,
            )
            self.reranker = CrossEncoder(RERANKER_MODEL, device=self.device)

        print(f"Catalog Search Service Ready! device={self.device}")

    def search(self, query: str, limit: int = 12, debug: bool = False) -> dict[str, Any]:
        start = time.time()
        intent = self.query_understanding.analyze(query)
        if not intent.rewritten_query:
            return self._empty_result(intent, start)

        docs = self.product_store.similarity_search(intent.rewritten_query, k=24)
        candidates = []
        for idx, doc in enumerate(docs):
            meta = self._resolve_doc_metadata(doc)
            if not meta.get("product_id"):
                continue
            if not self._passes_soft_filters(meta, intent):
                continue
            candidates.append(
                {
                    "idx": idx,
                    "doc": doc,
                    "meta": meta,
                    "context": self._resolve_doc_context(doc, meta),
                    "semantic_score": self._semantic_score_from_doc(doc),
                }
            )

        if not candidates:
            candidates = [
                {
                    "idx": idx,
                    "doc": doc,
                    "meta": self._resolve_doc_metadata(doc),
                    "context": self._resolve_doc_context(doc, self._resolve_doc_metadata(doc)),
                    "semantic_score": self._semantic_score_from_doc(doc),
                }
                for idx, doc in enumerate(docs)
                if self._resolve_doc_metadata(doc).get("product_id")
            ]

        candidates = candidates[:12]
        rerank_scores = self._rerank(intent.rewritten_query, candidates)

        scored = []
        for item, rerank_score in zip(candidates, rerank_scores):
            meta = item["meta"]
            entity_score = self._entity_boost(meta, intent)
            stock_score = self._stock_boost(meta)
            token_score = self._token_overlap_score(intent.rewritten_query, meta)
            image_score = self._visual_image_boost(intent, meta) if intent.visual_query else 0.0
            final_score = (
                (0.55 * float(rerank_score))
                + (0.20 * token_score)
                + (0.15 * entity_score)
                + (0.05 * stock_score)
                + (0.05 * image_score)
            )
            scored.append({**item, "rerank_score": float(rerank_score), "entity_score": entity_score, "stock_score": stock_score, "token_score": token_score, "image_score": image_score, "final_score": final_score})

        scored.sort(key=lambda item: item["final_score"], reverse=True)
        products = self._dedupe_products(scored, limit)

        response = {
            "normalized_query": intent.normalized_query,
            "rewritten_query": intent.rewritten_query,
            "intent": intent.intent,
            "filters": intent.filters,
            "products": products,
            "latency_ms": int((time.time() - start) * 1000),
        }
        if debug:
            response["search_debug"] = self._build_debug(scored)
        return response

    def _empty_result(self, intent: VoiceQueryIntent, start: float) -> dict[str, Any]:
        return {
            "normalized_query": intent.normalized_query,
            "rewritten_query": intent.rewritten_query,
            "intent": intent.intent,
            "filters": intent.filters,
            "products": [],
            "latency_ms": int((time.time() - start) * 1000),
            "search_debug": {"reason": "empty_query", "intent": asdict(intent)},
        }

    def _resolve_doc_metadata(self, doc) -> dict:
        meta = doc.metadata or {}
        if meta.get("product_id") is not None:
            return meta
        point_id = meta.get("_id")
        if point_id is None:
            return meta
        try:
            points = self.client.retrieve(
                collection_name=COLLECTION_PRODUCT_TEXT,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                return meta
            payload = points[0].payload or {}
            nested_meta = payload.get("metadata") if isinstance(payload, dict) else None
            return nested_meta if isinstance(nested_meta, dict) else payload
        except Exception:
            return meta

    def _resolve_doc_context(self, doc, meta: dict) -> str:
        content = (getattr(doc, "page_content", "") or "").strip()
        if content:
            return content
        return " ".join(
            str(part)
            for part in [
                meta.get("name"),
                meta.get("brand_name") or meta.get("brand"),
                meta.get("category_name") or meta.get("category"),
                meta.get("description"),
                meta.get("search_text"),
                meta.get("color_names"),
                meta.get("size_names"),
            ]
            if part
        )

    def _semantic_score_from_doc(self, doc) -> float:
        meta = doc.metadata or {}
        for key in ("score", "_score"):
            if key in meta:
                try:
                    return float(meta[key])
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def _passes_soft_filters(self, meta: dict, intent: VoiceQueryIntent) -> bool:
        filters = intent.filters
        if not filters:
            return True
        haystack = self._meta_haystack(meta)

        for key in ("category", "color", "gender", "size"):
            value = filters.get(key)
            if value and normalize_vietnamese_text(str(value)) not in haystack:
                if key in {"category", "gender"}:
                    return False
        price_filter = filters.get("price") or {}
        price = float(meta.get("price") or 0)
        if price_filter.get("min") and price < float(price_filter["min"]):
            return False
        if price_filter.get("max") and price > float(price_filter["max"]):
            return False
        return True

    def _rerank(self, query: str, candidates: list[dict]) -> list[float]:
        if not candidates:
            return []
        pairs = [[query, item["context"]] for item in candidates]
        try:
            scores = self.reranker.predict(pairs)
            return [float(score) for score in scores]
        except Exception as exc:
            print(f"Voice reranker failed, using token fallback: {exc}")
            return [self._token_overlap_score(query, item["meta"]) for item in candidates]

    def _entity_boost(self, meta: dict, intent: VoiceQueryIntent) -> float:
        filters = intent.filters
        if not filters:
            return 0.0
        haystack = self._meta_haystack(meta)
        checks = []
        for key in ("category", "color", "gender", "size"):
            value = filters.get(key)
            if value:
                checks.append(1.0 if normalize_vietnamese_text(str(value)) in haystack else 0.0)
        return sum(checks) / max(len(checks), 1)

    def _stock_boost(self, meta: dict) -> float:
        variants = meta.get("variant_summary") or meta.get("variants") or []
        if not isinstance(variants, list) or not variants:
            return 0.0
        for variant in variants:
            if isinstance(variant, dict) and int(variant.get("stock_quantity") or 0) > 0:
                return 1.0
        return 0.0

    def _visual_image_boost(self, intent: VoiceQueryIntent, meta: dict) -> float:
        if self.image_service is None:
            return 0.0
        try:
            hits = self.image_service.search_by_text_vector(
                intent.rewritten_query,
                limit=1,
                filter=qd_models.Filter(
                    must=[
                        qd_models.FieldCondition(
                            key="product_id",
                            match=qd_models.MatchValue(value=int(meta.get("product_id"))),
                        )
                    ]
                ),
            )
            if not hits:
                return 0.0
            return max(float(hits[0].score), 0.0)
        except Exception:
            return 0.0

    def _token_overlap_score(self, query: str, meta: dict) -> float:
        tokens = [t for t in normalize_vietnamese_text(query).split() if len(t) >= 2]
        if not tokens:
            return 0.0
        haystack = self._meta_haystack(meta)
        return sum(1 for token in tokens if token in haystack) / len(tokens)

    def _meta_haystack(self, meta: dict) -> str:
        return normalize_vietnamese_text(
            " ".join(
                str(part)
                for part in [
                    meta.get("name"),
                    meta.get("brand_name") or meta.get("brand"),
                    meta.get("category_name") or meta.get("category"),
                    meta.get("category_department"),
                    meta.get("description"),
                    meta.get("search_text"),
                    " ".join(meta.get("color_names") or []),
                    " ".join(meta.get("size_names") or []),
                ]
                if part
            )
        )

    def _dedupe_products(self, scored: list[dict], limit: int) -> list[dict[str, Any]]:
        products = []
        seen = set()
        for item in scored:
            meta = item["meta"]
            product_id = meta.get("product_id")
            if product_id in seen:
                continue
            products.append(
                {
                    "product_id": int(product_id),
                    "score": round(float(item["final_score"]), 6),
                    "image_url": meta.get("primary_image_url") or meta.get("image_url") or "",
                }
            )
            seen.add(product_id)
            if len(products) >= limit:
                break
        return products

    def _build_debug(self, scored: list[dict]) -> dict[str, Any]:
        return {
            "candidates": [
                {
                    "product_id": item["meta"].get("product_id"),
                    "name": item["meta"].get("name"),
                    "category": item["meta"].get("category_name") or item["meta"].get("category"),
                    "brand": item["meta"].get("brand_name") or item["meta"].get("brand"),
                    "final_score": round(float(item["final_score"]), 4),
                    "rerank_score": round(float(item["rerank_score"]), 4),
                    "entity_score": round(float(item["entity_score"]), 4),
                    "token_score": round(float(item["token_score"]), 4),
                    "stock_score": round(float(item["stock_score"]), 4),
                    "image_score": round(float(item["image_score"]), 4),
                }
                for item in scored[:12]
            ]
        }
