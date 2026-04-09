import os
import json
import re
import time
import unicodedata
from importlib import import_module
from importlib.util import find_spec
import numpy as np
import psycopg2
import torch
from dotenv import load_dotenv

# Libs
from langchain_groq import ChatGroq

if find_spec("langchain_huggingface"):
    HuggingFaceEmbeddings = import_module(
        "langchain_huggingface"
    ).HuggingFaceEmbeddings
else:
    HuggingFaceEmbeddings = import_module(
        "langchain_community.embeddings"
    ).HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from qdrant_client import models 

# App imports
from app.models.schemas import ProductSyncRequest, ProductInfo, ChatSessionState, ParsedChatQuery
from app.services.query_intent_service import QueryIntentService
from app.core.config import (
    TEXT_EMBEDDING_MODEL, 
    TEXT_VECTOR_SIZE,
    LLM_MODEL_NAME, 
    LLM_REWRITE_MODEL,
    RERANKER_MODEL,
    COLLECTION_PRODUCT_TEXT,
    COLLECTION_POLICIES,
    POSTGRES_CONFIG,
    COLLECTION_PRODUCT_IMAGE,
)

load_dotenv()

class RagService:
    REWRITE_PRONOUN_HINTS = {
        "no", "nay", "kia", "do", "item", "sp", "san", "pham",
        "it", "this", "that", "them", "those", "cai", "con", "mau", "size",
    }

    ANCHOR_STOPWORDS = {
        "shop", "fshop", "co", "khong", "ban", "toi", "minh", "xin", "cho",
        "mau", "size", "kich", "co", "bao", "nhieu", "nao", "la", "gi", "ve",
        "product", "san", "pham", "hang", "khong",
    }

    PRODUCT_ANCHOR_HINTS = {
        "jean", "jeans", "quan", "ao", "vay", "dam", "giay", "dep", "hoodie", "shirt", "balo",
        "tee", "polo", "jacket", "sweater", "short", "skirt", "dirtycoins", "converse",
        "nike", "adidas", "mlb", "new", "balance", "core", "vans",
    }

    DISCOVERY_KEYWORDS = {
        "co", "ban", "khong", "shop", "tim", "goi", "y", "mua", "muon",
        "recommend", "suggest", "sell", "have",
    }

    CORE_PRODUCT_NOUNS = {
        "quan", "ao", "vay", "dam", "giay", "dep", "hoodie", "jacket", "khoac", "balo",
        "shirt", "tee", "polo", "short", "skirt", "jean", "jeans",
    }

    def __init__(self, client):
        self.client = client
        self.query_intent_service = QueryIntentService()
        self.chat_debug_enabled = os.getenv("AI_CHAT_DEBUG", "true").strip().lower() in {"1", "true", "yes", "on"}

        # 1. Hardware Detection
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"Detected GPU (NVIDIA): {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            print("Detected GPU (Mac M-Series)")
        else:
            print("No GPU detected. Using CPU (Slow).")
        
        # 2. Embedding & Re-ranker
        print(f"🚀 Loading Models... (LLM: {LLM_MODEL_NAME})")
        self.embeddings = HuggingFaceEmbeddings(model_name=TEXT_EMBEDDING_MODEL, model_kwargs={"device": self.device})
        self.reranker = CrossEncoder(RERANKER_MODEL, device=self.device)

        self._ensure_collections_ready()
        
        # Khởi tạo Vector Store
        self.product_store = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_PRODUCT_TEXT,
            embedding=self.embeddings
        )
        self.policy_store = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_POLICIES,
            embedding=self.embeddings
        )
        
        # 3. LLM Setup
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("❌ WARNING: Cannot find GROQ_API_KEY in .env file")
            print("👉 Please create a .env file and add: GROQ_API_KEY=gsk_...")
            raise ValueError("Missing Groq API Key")

        # LLM CHÍNH
        self.llm_main = ChatGroq(
            groq_api_key=api_key,
            model_name=LLM_MODEL_NAME,
            temperature=0.3
        )

        # LLM PHỤ (Rewrite Query)
        self.llm_fast = ChatGroq(
            groq_api_key=api_key,
            model_name=LLM_REWRITE_MODEL,
            temperature=0.1
        )
        
        self._init_prompts()
        print("RAG Service Ready (Multi-Intent Supported)!")

    def _chat_log(self, message: str):
        if self.chat_debug_enabled:
            print(f"🧪 CHAT DEBUG | {message}")

    def _compact_meta(self, meta: dict) -> dict:
        return {
            "product_id": meta.get("product_id"),
            "name": meta.get("name"),
            "category": meta.get("category_name") or meta.get("category"),
            "brand": meta.get("brand_name") or meta.get("brand"),
            "price": meta.get("price"),
        }

    def _ensure_collections_ready(self):
        """Create required text collections if they are missing to avoid startup failure."""
        required_collections = [COLLECTION_PRODUCT_TEXT, COLLECTION_POLICIES]
        for collection_name in required_collections:
            try:
                if self.client.collection_exists(collection_name):
                    continue

                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=TEXT_VECTOR_SIZE,
                        distance=models.Distance.COSINE,
                    ),
                )
                print(f"ℹ️ Created missing Qdrant collection: {collection_name}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to ensure collection '{collection_name}': {e}"
                ) from e
    
    def _init_prompts(self):
        self.query_parser_prompt = ChatPromptTemplate.from_template("""
        Bạn là bộ phân tích truy vấn có cấu trúc cho chatbot thời trang.
        Nhiệm vụ của bạn là trả về JSON hợp lệ duy nhất, không dùng markdown, không giải thích.

        Session State hiện tại:
        {session_state}

        History gần nhất:
        {history}

        User Question:
        {question}

        Hãy trả về đúng schema sau:
        {{
            "intent": "product_search" | "size_advice" | "color_question" | "policy" | "order" | "unknown",
            "search_query": "chuỗi truy vấn ngắn gọn để tìm trong catalog",
            "requires_context": boolean,
            "confidence": number,
            "follow_up": boolean,
            "entities": {{
                "product_name": string | null,
                "brand": string | null,
                "category": string | null,
                "color": string | null,
                "size": string | null,
                "gender": string | null,
                "height_cm": number | null,
                "weight_kg": number | null
            }}
        }}

        Quy tắc:
        - Nếu là câu follow-up về size/màu và session state có sản phẩm đang active, giữ sản phẩm đó trong search_query.
        - Nếu user nhắc đến sản phẩm mới rõ ràng, search_query phải bám sản phẩm mới đó.
        - Không được trả lời người dùng, chỉ xuất JSON.
        """)

        self.query_parser_fallback_prompt = ChatPromptTemplate.from_template("""
        Bạn là bộ phân tích truy vấn có cấu trúc cho chatbot thời trang.
        Trả về JSON hợp lệ duy nhất, không dùng markdown, không giải thích.

        Session State hiện tại:
        {session_state}

        History gần nhất:
        {history}

        User Question:
        {question}

        Schema bắt buộc:
        {{
            "intent": "product_search" | "size_advice" | "color_question" | "policy" | "order" | "unknown",
            "search_query": "chuỗi truy vấn ngắn gọn để tìm trong catalog",
            "requires_context": boolean,
            "confidence": number,
            "follow_up": boolean,
            "entities": {{
                "product_name": string | null,
                "brand": string | null,
                "category": string | null,
                "color": string | null,
                "size": string | null,
                "gender": string | null,
                "height_cm": number | null,
                "weight_kg": number | null
            }}
        }}

        Chỉ xuất JSON.
        """)

        # 1. ROUTER PROMPT
        self.router_prompt = ChatPromptTemplate.from_template("""
        Phân loại ý định của câu hỏi người dùng vào 3 nhóm: ORDER, POLICY, PRODUCT.

        - ORDER: Các câu hỏi về đơn hàng, trạng thái, theo dõi, lịch sử mua, hủy đơn, chi tiết đơn cụ thể, ...
        - POLICY: Các câu hỏi về chính sách của shop (phí ship, đổi trả, thanh toán, địa chỉ, thông tin của hàng).
        - PRODUCT: Các câu hỏi tìm sản phẩm, gợi ý, giá, tồn kho, màu sắc, kích cỡ.

        Câu hỏi: {question}

        CHỈ TRẢ VỀ MỘT TỪ: ORDER, POLICY, hoặc PRODUCT.
        """)

        # A. Prompt viết lại câu hỏi tìm kiếm
        self.rewrite_prompt = ChatPromptTemplate.from_template("""
        Bạn là bộ tối ưu truy vấn tìm kiếm cho cơ sở dữ liệu thời trang.
        Viết lại câu hỏi thành truy vấn ngắn gọn, rõ nghĩa, ưu tiên tiếng Việt.
        Giữ nguyên tên thương hiệu/tên sản phẩm gốc nếu cần.
        Loại bỏ từ thừa và giải quyết đại từ dựa vào History.

        History: {history}
        User Question: {question}

        CHỈ TRẢ VỀ TRUY VẤN ĐÃ VIẾT LẠI. KHÔNG GIẢI THÍCH.
        """)

        self.rewrite_strict_prompt = ChatPromptTemplate.from_template("""
        NHIỆM VỤ: Tạo truy vấn tìm kiếm ngắn gọn cho hệ thống RAG thời trang.

        QUY TẮC BẮT BUỘC:
        - Chỉ trả về MỘT DÒNG DUY NHẤT.
        - KHÔNG được trả lời dạng hội đáp, KHÔNG giải thích, KHÔNG xưng hô tôi/bạn.
        - Nếu câu hiện tại phụ thuộc ngữ cảnh ("size nào", "màu nào", "còn không"), phải bổ sung tên sản phẩm từ History.
        - Giữ lại thông tin quan trọng: tên sản phẩm/thương hiệu, màu, size, cân nặng, chiều cao.

        History: {history}
        User Question: {question}

        CHỈ TRẢ VỀ TRUY VẤN.
        """)

        # B. Prompt trả lời về Sản phẩm
        self.product_qa_prompt = ChatPromptTemplate.from_template("""
        Bạn là trợ lý bán hàng thời trang của "FShop".
        
        RECOMMENDED PRODUCTS:
        {context}
        
        USER QUESTION: {question}

        INSTRUCTIONS:
        1. Luôn trả lời bằng tiếng Việt có dấu, tự nhiên, dễ hiểu.
        2. Chỉ dựa vào danh sách sản phẩm ở trên.
        3. Nếu tìm thấy sản phẩm, nêu tên và giá rõ ràng.
        4. Nếu context rỗng, xin lỗi lịch sự và nói rõ không tìm thấy.
        """)

        # C. Prompt trả lời về Chính sách
        self.policy_qa_prompt = ChatPromptTemplate.from_template("""
        Bạn là nhân viên chăm sóc khách hàng.
        Trả lời dựa DUY NHẤT vào phần chính sách bên dưới.
        
        POLICY CONTEXT:
        {context}
        
        USER QUESTION: {question}
        
        Trả lời ngắn gọn, chuyên nghiệp, bằng tiếng Việt có dấu.
        """)

        self.order_qa_prompt = ChatPromptTemplate.from_template("""
        Bạn là nhân viên chăm sóc khách hàng thân thiện.
        
        USER ORDER DATA:
        {order_context}
        
        USER QUESTION: {question}
        
        INSTRUCTIONS:
        1. Chỉ trả lời dựa trên dữ liệu đơn hàng đã cung cấp.
        2. Nếu status là 'PENDING', nói đơn đang được đóng gói.
        3. Nếu status là 'SHIPPED', nói đơn đang trên đường giao.
        4. Nếu status là 'DELIVERED', gửi lời chúc khách sử dụng vui vẻ.
        5. Nếu người dùng muốn hủy, hướng dẫn kiểm tra app/website (thường chỉ hủy được PENDING).
        6. Nếu không có đơn, nhắc người dùng kiểm tra Order ID.
        7. Trả lời bằng tiếng Việt có dấu.
        """)

    def _build_chat_result(
        self,
        answer: str,
        products: list[ProductInfo],
        session_state: ChatSessionState | None = None,
        parsed_query: ParsedChatQuery | None = None,
    ) -> dict:
        return {
            "answer": answer,
            "products": products,
            "session_state": session_state.model_dump() if session_state else None,
            "parsed_query": parsed_query.model_dump() if parsed_query else None,
        }

    def _serialize_session_state(self, session_state: ChatSessionState | dict | None) -> str:
        if not session_state:
            return "{}"
        if isinstance(session_state, ChatSessionState):
            return json.dumps(session_state.model_dump(), ensure_ascii=False)
        return json.dumps(session_state, ensure_ascii=False)

    def _extract_json_payload(self, text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return ""

        code_fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
        if code_fence_match:
            raw = code_fence_match.group(1).strip()

        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return raw
        return raw[start:end + 1]

    def _normalize_parser_intent(self, intent: str) -> str:
        normalized = self._normalize_text(intent)
        if normalized in {"order", "don hang"}:
            return "order"
        if normalized in {"policy", "chinh sach"}:
            return "policy"
        if normalized in {"size advice", "size", "kich co"}:
            return "size_advice"
        if normalized in {"color question", "color", "mau sac"}:
            return "color_question"
        if normalized in {"product search", "product", "san pham"}:
            return "product_search"
        return normalized if normalized in {"product_search", "size_advice", "color_question", "policy", "order", "unknown"} else "unknown"

    def _default_parsed_query(self, question: str) -> ParsedChatQuery:
        return ParsedChatQuery(
            intent="product_search",
            search_query=question,
            requires_context=False,
            confidence=0.0,
            entities={},
            follow_up=False,
        )

    def _validate_parsed_query(self, parsed: ParsedChatQuery, question: str) -> ParsedChatQuery | None:
        if not parsed.search_query or not parsed.search_query.strip():
            parsed.search_query = question

        parsed.intent = self._normalize_parser_intent(parsed.intent)
        if parsed.intent == "unknown":
            parsed.intent = "product_search"

        parsed.search_query = parsed.search_query.strip()
        parsed.entities = parsed.entities or {}
        return parsed

    def _parse_chat_query(self, question: str, history: list, session_state: ChatSessionState | dict | None) -> ParsedChatQuery:
        history_text = "\n".join([f"{msg.role}: {msg.content}" for msg in history[-8:]]) if history else ""
        session_text = self._serialize_session_state(session_state)

        chain = self.query_parser_prompt | self.llm_fast | StrOutputParser()
        fallback_chain = self.query_parser_fallback_prompt | self.llm_main | StrOutputParser()

        for current_chain in (chain, fallback_chain):
            try:
                raw = current_chain.invoke({"session_state": session_text, "history": history_text, "question": question})
                payload = json.loads(self._extract_json_payload(raw))
                parsed = ParsedChatQuery.model_validate(payload)
                parsed = self._validate_parsed_query(parsed, question)
                if parsed and parsed.confidence >= 0.35:
                    return parsed
            except Exception as e:
                print(f"Parser fallback error: {e}")

        return self._default_parsed_query(question)

    def _build_state_from_meta(
        self,
        parsed_query: ParsedChatQuery,
        meta: dict | None,
        previous_state: ChatSessionState | dict | None,
    ) -> ChatSessionState:
        previous = previous_state if isinstance(previous_state, ChatSessionState) else ChatSessionState.model_validate(previous_state or {})

        if not meta:
            return ChatSessionState(
                active_product_id=previous.active_product_id,
                active_product_name=previous.active_product_name,
                active_category=previous.active_category,
                active_brand=previous.active_brand,
                last_intent=parsed_query.intent,
                last_entities=parsed_query.entities,
            )

        active_product_id = int(meta.get("product_id", 0) or 0) or previous.active_product_id
        active_product_name = meta.get("name") or previous.active_product_name
        active_category = meta.get("category_name") or meta.get("category") or previous.active_category
        active_brand = meta.get("brand_name") or meta.get("brand") or previous.active_brand

        if parsed_query.intent in {"size_advice", "color_question"} and previous.active_product_id:
            active_product_id = previous.active_product_id
            active_product_name = previous.active_product_name or active_product_name
            active_category = previous.active_category or active_category
            active_brand = previous.active_brand or active_brand

        return ChatSessionState(
            active_product_id=active_product_id,
            active_product_name=active_product_name,
            active_category=active_category,
            active_brand=active_brand,
            last_intent=parsed_query.intent,
            last_entities=parsed_query.entities,
        )

    def _default_state_from_query(
        self,
        parsed_query: ParsedChatQuery,
        previous_state: ChatSessionState | dict | None,
    ) -> ChatSessionState:
        previous = previous_state if isinstance(previous_state, ChatSessionState) else ChatSessionState.model_validate(previous_state or {})
        return ChatSessionState(
            active_product_id=previous.active_product_id,
            active_product_name=previous.active_product_name,
            active_category=previous.active_category,
            active_brand=previous.active_brand,
            last_intent=parsed_query.intent,
            last_entities=parsed_query.entities,
        )

    def _get_user_orders_sql(self, user_id: int, specific_order_id: str = None):
        """
        Query trực tiếp DB để lấy thông tin đơn hàng mới nhất
        """
        conn = None
        try:
            conn = psycopg2.connect(**POSTGRES_CONFIG)
            cur = conn.cursor()
            
            query = """
                SELECT 
                    o.id, 
                    o.status, 
                    o.total_amount,
                    o.created_at,
                    string_agg(DISTINCT CONCAT(p.name, ' (x', oi.quantity, ')'), ', ') as items
                FROM orders o
                LEFT JOIN order_items oi ON o.id = oi.order_id
                LEFT JOIN product_variants pv ON oi.variant_id = pv.id
                LEFT JOIN products p ON pv.product_id = p.id
                WHERE o.user_id = %s
            """
            
            params = [user_id]
            
            if specific_order_id:
                query += " AND o.id = %s"
                params.append(specific_order_id)
            
            query += ' GROUP BY o.id ORDER BY o.created_at DESC LIMIT 3'
            
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            
            if not rows:
                return "Không tìm thấy đơn hàng nào."

            order_text = ""
            for row in rows:
                oid, status, total, created_at, items = row
                order_text += (
                    f"- Order #{oid}\n"
                    f"  Status: {status}\n"
                    f"  Total: ${total}\n"
                    f"  Date: {created_at}\n"
                    f"  Items: {items}\n\n"
                )
            return order_text
            
        except Exception as e:
            print(f"❌ DB Error: {e}")
            return "Có lỗi khi lấy dữ liệu đơn hàng."
        finally:
            if conn: conn.close()
        

    def _detect_intent_fast(self, question: str) -> str:
        """
        Phân loại ý định cực nhanh bằng từ khóa (Keyword Spotting).
        """
        q = self._normalize_text(question)

        order_keywords = [
            "order", "track", "package", "delivery", "status", "shipment", "cancel", "bought", "purchase history",
            "don hang", "ma don", "theo doi", "van don", "trang thai", "giao hang", "huy don", "lich su mua"
        ]
        if any(kw in q for kw in order_keywords):
            return "ORDER"
        
        policy_keywords = [
            "policy", "shipping", "ship", "delivery", "deliver",
            "return", "refund", "exchange", "money back",
            "payment", "pay", "credit card", "cod", "cash",
            "address", "location", "store", "shop open", "hour", "contact", "phone",
            "chinh sach", "phi ship", "van chuyen", "doi tra", "hoan tien", "thanh toan",
            "dia chi", "cua hang", "gio mo cua", "lien he", "so dien thoai", "kich co", "bang size"
        ]
        
        if any(kw in q for kw in policy_keywords):
            return "POLICY"
        
        return "PRODUCT"
    
    def _rewrite_query(self, question: str, history: list) -> str:
        if not history:
            return question
        if not self._should_rewrite_query(question):
            return question
            
        start_time = time.time()
        history_text = "\n".join([f"{msg.role}: {msg.content}" for msg in history[-6:]])
        chain = self.rewrite_prompt | self.llm_fast | StrOutputParser()
        try:
            new_query = chain.invoke({"history": history_text, "question": question})
            clean_query = self._sanitize_rewrite(new_query)

            if not self._is_valid_rewrite(clean_query, question):
                strict_chain = self.rewrite_strict_prompt | self.llm_fast | StrOutputParser()
                strict_query = strict_chain.invoke({"history": history_text, "question": question})
                strict_clean_query = self._sanitize_rewrite(strict_query)

                if self._is_valid_rewrite(strict_clean_query, question):
                    print(
                        f"⚡ Rewritten(strict) ({time.time() - start_time:.2f}s): "
                        f"'{question}' -> '{strict_clean_query}'"
                    )
                    return strict_clean_query

                print(
                    "⚠️ Rewrite rejected, fallback to history-aware query. "
                    f"candidate='{clean_query[:120]}' strict='{strict_clean_query[:120]}'"
                )
                return self._build_history_aware_query(question, history)

            print(f"⚡ Rewritten ({time.time() - start_time:.2f}s): '{question}' -> '{clean_query}'")
            return clean_query
        except Exception as e:
            print(f"Rewrite Error: {e}")
            return self._build_history_aware_query(question, history)

    def _message_role(self, msg) -> str:
        if isinstance(msg, dict):
            return str(msg.get("role", "")).strip().lower()
        return str(getattr(msg, "role", "")).strip().lower()

    def _message_content(self, msg) -> str:
        if isinstance(msg, dict):
            return str(msg.get("content", "") or "")
        return str(getattr(msg, "content", "") or "")

    def _find_recent_product_context(self, history: list) -> str:
        for msg in reversed(history[-8:]):
            content = self._message_content(msg)
            role = self._message_role(msg)
            if role not in {"user", "assistant"}:
                continue
            anchor_tokens = self._extract_anchor_tokens(content)
            if not anchor_tokens:
                continue
            if any(token in self.PRODUCT_ANCHOR_HINTS for token in anchor_tokens):
                return content.strip()
        return ""

    def _build_history_aware_query(self, question: str, history: list) -> str:
        if not self._is_context_dependent_followup(question):
            return question

        recent_product_context = self._find_recent_product_context(history)
        if not recent_product_context:
            return question
        # Keep query compact but preserve product anchor from previous turn.
        return f"{question}. San pham dang duoc hoi: {recent_product_context}"

    def _is_context_dependent_followup(self, question: str) -> bool:
        normalized = self._normalize_text(question)
        if not normalized:
            return False

        anchor_tokens = self._extract_anchor_tokens(question)
        has_product_anchor = any(token in self.PRODUCT_ANCHOR_HINTS for token in anchor_tokens)

        # Coreference-like terms indicate follow-up questions.
        if re.search(r"\b(no|nay|kia|do|this|that|it|them|those|cai|con)\b", normalized):
            return True

        # Size/color question without explicit product should reuse previous product context.
        if (self._is_size_question(question) or self._is_color_question(question)) and not has_product_anchor:
            return True

        return False

    def _is_explicit_product_query(self, question: str) -> bool:
        if self._is_context_dependent_followup(question):
            return False

        anchor_tokens = self._extract_anchor_tokens(question)
        if not anchor_tokens:
            return False

        product_anchor_tokens = [tok for tok in anchor_tokens if tok in self.PRODUCT_ANCHOR_HINTS]
        if not product_anchor_tokens:
            return False

        meaningful_tokens = [tok for tok in anchor_tokens if tok not in self.DISCOVERY_KEYWORDS]
        return len(meaningful_tokens) > 0

    def _sanitize_rewrite(self, text: str) -> str:
        normalized = (text or "").replace('"', " ").replace("'", " ")
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = re.sub(r"^(assistant|user|query|truy van)\s*:\s*", "", normalized, flags=re.IGNORECASE)
        return normalized

    def _should_rewrite_query(self, question: str) -> bool:
        normalized = self._normalize_text(question)
        if not normalized:
            return False

        anchor_tokens = self._extract_anchor_tokens(question)
        has_product_anchor = any(token in self.PRODUCT_ANCHOR_HINTS for token in anchor_tokens)

        # Very short queries without product anchors usually need context from history.
        if len(anchor_tokens) <= 2 and not has_product_anchor:
            return True

        # Rewrite when question is context-dependent and lacks explicit product anchors.
        pronoun_pattern = r"\b(no|nay|kia|do|this|that|it|them|those|cai|con)\b"
        size_color_followup = r"\b(size|mau|co con|con hang|bao nhieu|nao|the nao|sao)\b"
        if re.search(pronoun_pattern, normalized):
            return True

        if re.search(size_color_followup, normalized) and not has_product_anchor:
            return True

        return False

    def _extract_anchor_tokens(self, text: str) -> list[str]:
        normalized = self._normalize_text(text)
        tokens = []
        for token in normalized.split():
            if len(token) < 3:
                continue
            if token in self.ANCHOR_STOPWORDS:
                continue
            if token in self.REWRITE_PRONOUN_HINTS:
                continue
            tokens.append(token)
        return tokens

    def _is_valid_rewrite(self, rewritten: str, original: str) -> bool:
        text = (rewritten or "").strip()
        if not text:
            return False

        lowered = text.lower()

        # Guard against answer-like generations.
        if lowered.startswith("assistant:") or lowered.startswith("user:"):
            return False
        normalized_rewrite = self._normalize_text(text)

        # Reject SQL-like rewrites generated by the LLM.
        if re.search(r"\b(select|from|where|join|group by|order by|insert|update|delete)\b", normalized_rewrite):
            return False

        if any(phrase in normalized_rewrite for phrase in [
            "xin chào", "tôi hiểu", "tôi có thể", "hy vọng", "cảm ơn", "xin lỗi",
            "vui lòng", "nếu bạn cần", "theo thông tin", "mình đề xuất", "có các size sau"
        ]):
            return False

        # Reject list-like answer content instead of compact search query.
        if re.search(r"\b(xs|s|m|l|xl|xxl|xxxl)\s*:\s*\d", lowered):
            return False

        if len(text) > 260:
            return False

        original_len = len(self._normalize_text(original).split())
        rewritten_len = len(normalized_rewrite.split())
        if original_len > 0 and rewritten_len > max(original_len + 6, int(original_len * 1.6)):
            return False

        # Rewritten query should keep product anchors from original question.
        original_tokens = self._extract_anchor_tokens(original)
        rewritten_norm = normalized_rewrite
        if original_tokens and not any(tok in rewritten_norm for tok in original_tokens):
            return False

        return True

    def _score_doc_anchor(self, query: str, meta: dict) -> float:
        query_tokens = self._extract_anchor_tokens(query)
        if not query_tokens:
            return 0.0

        haystack = self._normalize_text(
            " ".join(
                [
                    str(meta.get("name", "")),
                    str(meta.get("brand_name") or meta.get("brand", "")),
                    str(meta.get("category_name") or meta.get("category", "")),
                    str(meta.get("description", "")),
                ]
            )
        )
        if not haystack:
            return 0.0

        hits = sum(1 for token in query_tokens if token in haystack)
        return hits / max(len(query_tokens), 1)

    def _extract_core_product_nouns(self, query: str) -> list[str]:
        tokens = self._normalize_text(query).split()
        return [tok for tok in tokens if tok in self.CORE_PRODUCT_NOUNS]

    def _meta_matches_core_nouns(self, meta: dict, core_nouns: list[str]) -> bool:
        if not core_nouns:
            return True

        haystack = self._normalize_text(
            " ".join(
                [
                    str(meta.get("name", "")),
                    str(meta.get("category_name") or meta.get("category", "")),
                    str(meta.get("description", "")),
                ]
            )
        )
        if not haystack:
            return False

        hits = sum(1 for noun in core_nouns if noun in haystack)
        required_hits = 2 if len(core_nouns) >= 2 else 1
        return hits >= required_hits

    def _has_explicit_product_signal(self, parsed_query: ParsedChatQuery) -> bool:
        entities = parsed_query.entities or {}
        if entities.get("product_name"):
            return True
        if entities.get("category"):
            return True
        if entities.get("brand") and entities.get("product_name"):
            return True
        return False

    def _should_use_previous_context(self, parsed_query: ParsedChatQuery) -> bool:
        if parsed_query.follow_up:
            return True
        if not parsed_query.requires_context:
            return False
        # A product_search without follow_up means the user is asking about a new/different
        # product. Appending previous session context (e.g. old product name/brand) would
        # pollute the retrieval query and return wrong results.
        if parsed_query.intent == "product_search":
            return False
        return not self._has_explicit_product_signal(parsed_query)

    def _is_alternative_query(self, question: str) -> bool:
        q = self._normalize_text(question)
        patterns = [
            r"\bngoai\b.*\b(con|co)\b.*\b(loai nao|khac|khong)\b",
            r"\bkhac\b.*\bkhong\b",
            r"\bcon\b.*\bloai nao\b",
        ]
        return any(re.search(pattern, q) for pattern in patterns)

    def _is_size_question(self, question: str) -> bool:
        q = self._normalize_text(question)
        size_keywords = [
            "size", "kich co", "co size", "bao nhieu size", "size nao", "bang size", "so do size"
        ]
        return any(k in q for k in size_keywords)

    def _is_color_question(self, question: str) -> bool:
        q = self._normalize_text(question)
        color_keywords = [
            "mau", "mau sac", "bao nhieu mau", "co may mau", "color", "colors"
        ]
        return any(k in q for k in color_keywords)

    def _extract_size_names(self, meta: dict) -> list[str]:
        sizes = set()

        raw_sizes = meta.get("size_names") or []
        if isinstance(raw_sizes, list):
            for size in raw_sizes:
                if size:
                    sizes.add(str(size).strip())

        variants = meta.get("variants") or []
        if isinstance(variants, list):
            for variant in variants:
                if not isinstance(variant, dict):
                    continue
                size_name = variant.get("size_name")
                if size_name:
                    sizes.add(str(size_name).strip())

        variant_summary = meta.get("variant_summary") or []
        if isinstance(variant_summary, list):
            for variant in variant_summary:
                if not isinstance(variant, dict):
                    continue
                size_name = variant.get("size_name")
                if size_name:
                    sizes.add(str(size_name).strip())

        return sorted([s for s in sizes if s], key=lambda s: (len(s), s))

    def _extract_color_names(self, meta: dict) -> list[str]:
        colors = set()

        raw_colors = meta.get("color_names") or []
        if isinstance(raw_colors, list):
            for color in raw_colors:
                if color:
                    colors.add(str(color).strip())

        variants = meta.get("variants") or []
        if isinstance(variants, list):
            for variant in variants:
                if not isinstance(variant, dict):
                    continue
                color_name = variant.get("color_name")
                if color_name:
                    colors.add(str(color_name).strip())

        variant_summary = meta.get("variant_summary") or []
        if isinstance(variant_summary, list):
            for variant in variant_summary:
                if not isinstance(variant, dict):
                    continue
                color_name = variant.get("color_name")
                if color_name:
                    colors.add(str(color_name).strip())

        return sorted([c for c in colors if c], key=lambda c: (len(c), c))

    def _build_size_answer(self, question: str, final_docs: list[Document]) -> str | None:
        if not self._is_size_question(question):
            return None

        weight_kg, height_cm = self._extract_body_metrics(question)

        for doc in final_docs:
            meta = self._resolve_doc_metadata(doc)
            size_names = self._extract_size_names(meta)
            if not size_names:
                continue

            product_name = meta.get("name", "sản phẩm")
            sizes_text = ", ".join(size_names)

            if weight_kg or height_cm:
                recommended_size = self._recommend_size_from_metrics(size_names, weight_kg, height_cm)
                metrics_parts = []
                if weight_kg:
                    metrics_parts.append(f"{weight_kg}kg")
                if height_cm:
                    metrics_parts.append(f"{height_cm}cm")
                metrics_text = " / ".join(metrics_parts)

                return (
                    f"{product_name} hiện có các size: {sizes_text}. "
                    f"Với thông tin {metrics_text}, mình đề xuất bạn thử size {recommended_size}. "
                    "Nếu bạn thích mặc rộng hơn, có thể tăng lên 1 size."
                )

            return (
                f"{product_name} hiện đang có các size: {sizes_text}. "
                "Bạn cần mình gợi ý size phù hợp theo chiều cao/cân nặng không?"
            )

        return None

    def _extract_body_metrics(self, question: str) -> tuple[int | None, int | None]:
        q = (question or "").lower()

        weight = None
        height_cm = None

        weight_match = re.search(r"(\d{2,3})\s*kg", q)
        if weight_match:
            try:
                weight = int(weight_match.group(1))
            except ValueError:
                weight = None

        height_cm_match = re.search(r"(\d{3})\s*cm", q)
        if height_cm_match:
            try:
                height_cm = int(height_cm_match.group(1))
            except ValueError:
                height_cm = None
        else:
            height_m_match = re.search(r"(1[.,]?\d{1,2})\s*m", q)
            if height_m_match:
                raw_h = height_m_match.group(1).replace(",", ".")
                try:
                    h_float = float(raw_h)
                    height_cm = int(round(h_float * 100))
                except ValueError:
                    height_cm = None
            else:
                compact_match = re.search(r"1m(\d{2})", q)
                if compact_match:
                    try:
                        height_cm = 100 + int(compact_match.group(1))
                    except ValueError:
                        height_cm = None

        return weight, height_cm

    def _size_rank(self, size_name: str) -> int:
        s = (size_name or "").strip().upper()
        mapping = {
            "XS": 0,
            "S": 1,
            "M": 2,
            "L": 3,
            "XL": 4,
            "XXL": 5,
            "XXXL": 6,
        }
        return mapping.get(s, 2)

    def _recommend_size_from_metrics(self, size_names: list[str], weight_kg: int | None, height_cm: int | None) -> str:
        target_rank = 2  # Default M

        if weight_kg is not None:
            if weight_kg < 55:
                target_rank = min(target_rank, 1)
            elif weight_kg < 65:
                target_rank = max(target_rank, 2)
            elif weight_kg < 75:
                target_rank = max(target_rank, 3)
            elif weight_kg < 85:
                target_rank = max(target_rank, 4)
            else:
                target_rank = max(target_rank, 5)

        if height_cm is not None:
            if height_cm < 165:
                target_rank = min(target_rank, 1)
            elif height_cm < 172:
                target_rank = max(target_rank, 2)
            elif height_cm < 178:
                target_rank = max(target_rank, 3)
            elif height_cm < 184:
                target_rank = max(target_rank, 4)
            else:
                target_rank = max(target_rank, 5)

        ranked_sizes = [(name, self._size_rank(name)) for name in size_names]
        ranked_sizes.sort(key=lambda item: abs(item[1] - target_rank))
        return ranked_sizes[0][0] if ranked_sizes else "M"

    def _build_color_answer(self, question: str, final_docs: list[Document]) -> str | None:
        if not self._is_color_question(question):
            return None

        for doc in final_docs:
            meta = self._resolve_doc_metadata(doc)
            color_names = self._extract_color_names(meta)
            if not color_names:
                continue

            product_name = meta.get("name", "sản phẩm")
            colors_text = ", ".join(color_names)
            return (
                f"{product_name} hiện đang có {len(color_names)} màu: {colors_text}. "
                "Bạn muốn mình gợi ý màu phù hợp theo phong cách của bạn không?"
            )

        return None

    def _resolve_doc_metadata(self, doc: Document) -> dict:
        """
        LangChain Qdrant sometimes returns only {_id, _collection_name} metadata
        when payload key conventions do not match. Resolve payload from Qdrant by id.
        """
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
            if isinstance(nested_meta, dict):
                return nested_meta
            if isinstance(payload, dict):
                return payload
            return meta
        except Exception as e:
            print(f"⚠️ Cannot resolve payload for point {point_id}: {e}")
            return meta

    def _resolve_doc_context(self, doc: Document, meta: dict) -> str:
        content = (doc.page_content or "").strip()
        if content:
            return content

        search_text = str(meta.get("search_text") or "").strip()
        if search_text:
            return search_text

        size_names = self._extract_size_names(meta)
        color_names = self._extract_color_names(meta)
        sizes_text = ", ".join(size_names) if size_names else ""
        colors_text = ", ".join(color_names) if color_names else ""

        return (
            f"Product: {meta.get('name', '')}. "
            f"Brand: {meta.get('brand_name') or meta.get('brand', '')}. "
            f"Category: {meta.get('category_name') or meta.get('category', '')}. "
            f"Department: {meta.get('category_department', '')}. "
            f"Description: {meta.get('description', '')}. "
            f"Sizes: {sizes_text}. "
            f"Colors: {colors_text}. "
            f"Price: {meta.get('price', 0)}."
        )

    def _normalize_text(self, text: str) -> str:
        text = (text or "").lower()
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _token_overlap_score(self, query: str, meta: dict) -> float:
        normalized_query = self._normalize_text(query)
        query_tokens = [t for t in normalized_query.split() if len(t) >= 2]
        if not query_tokens:
            return 0.0

        haystack = self._normalize_text(
            " ".join(
                [
                    str(meta.get("name", "")),
                    str(meta.get("category_name") or meta.get("category", "")),
                    str(meta.get("description", "")),
                ]
            )
        )
        hits = sum(1 for token in query_tokens if token in haystack)
        return hits / len(query_tokens)

    def _select_primary_product_index(
        self,
        user_query: str,
        search_query: str,
        scored_items: list[tuple[int, float, float, bool]],
        resolved_metas: list[dict],
    ) -> int | None:
        if not scored_items:
            return None

        best_idx = scored_items[0][0]
        best_score = -1.0

        for idx, boosted_score, anchor_score, is_match in scored_items:
            meta = resolved_metas[idx]
            name_score = self._score_doc_anchor(user_query, meta)
            rewrite_name_score = self._score_doc_anchor(search_query, meta)
            exact_name_bonus = 0.15 if self._normalize_text(user_query) in self._normalize_text(str(meta.get("name", ""))) else 0.0
            combined = float(boosted_score) + (0.50 * max(name_score, rewrite_name_score, anchor_score)) + exact_name_bonus + (0.10 if is_match else 0.0)

            if combined > best_score:
                best_score = combined
                best_idx = idx

        return best_idx

    async def chat(self, user_query: str, history: list, user_id: int = None, session_state: ChatSessionState | dict | None = None):
        total_start = time.time()
        print(f"User  asking: {user_query}")

        parsed_query = self._parse_chat_query(user_query, history, session_state)
        intent = parsed_query.intent
        print(f"🧭 Parsed Intent: {intent} | confidence={parsed_query.confidence:.2f}")
        self._chat_log(
            f"parsed_query={parsed_query.model_dump()} | session_state={session_state.model_dump() if isinstance(session_state, ChatSessionState) else session_state}"
        )

        # CASE A: ORDER 
        if intent == "ORDER":
            if not user_id:
                return self._build_chat_result("Vui lòng đăng nhập để kiểm tra đơn hàng.", [], None, parsed_query)
            
            # Trích xuất mã đơn hàng nếu có (Ví dụ: "Order 123")
            order_id_match = re.search(r'(?:order|#)\s*(\d+)', user_query.lower())
            specific_id = order_id_match.group(1) if order_id_match else None
            
            # Lấy data SQL
            order_data = self._get_user_orders_sql(user_id, specific_id)
            
            # Sinh câu trả lời
            chain = self.order_qa_prompt | self.llm_main | StrOutputParser()
            answer = chain.invoke({"order_context": order_data, "question": user_query})
            
            print(f"⏱️ TOTAL TIME: {time.time() - total_start:.2f}s")
            return self._build_chat_result(answer, [], None, parsed_query)

        # CASE B: POLICY
        elif intent == "POLICY":
            t0 = time.time()
            docs = self.policy_store.similarity_search(user_query, k=3)
            print(f"🔍 Policy Retrieval: {time.time() - t0:.2f}s")
            
            if not docs:
                return self._build_chat_result("Xin lỗi, tôi không tìm thấy thông tin chính sách phù hợp.", [], None, parsed_query)
            
            context_text = "\n\n".join([d.page_content for d in docs])
            chain = self.policy_qa_prompt | self.llm_main | StrOutputParser()
            answer = chain.invoke({"context": context_text, "question": user_query})
            return self._build_chat_result(answer, [], None, parsed_query)

        # CASE C: PRODUCT (Default)
        else:
            return await self._handle_product_search(user_query, history, parsed_query, session_state)

    async def _handle_product_search(self, user_query: str, history: list, parsed_query: ParsedChatQuery, session_state: ChatSessionState | dict | None):
        search_query = parsed_query.search_query or user_query
        use_previous_context = self._should_use_previous_context(parsed_query)
        if use_previous_context and session_state:
            state_obj = session_state if isinstance(session_state, ChatSessionState) else ChatSessionState.model_validate(session_state)
            if state_obj.active_product_name:
                search_query = f"{search_query} {state_obj.active_product_name}"
            if state_obj.active_brand:
                search_query = f"{search_query} {state_obj.active_brand}"

        self._chat_log(
            f"search_query='{search_query}' | intent={parsed_query.intent} | requires_context={parsed_query.requires_context} | follow_up={parsed_query.follow_up} | use_previous_context={use_previous_context}"
        )

        # 2. Retrieval
        t0 = time.time()
        docs = self.product_store.similarity_search(search_query, k=24)
        print(f"🔍 Product Retrieval: {time.time() - t0:.2f}s (Found {len(docs)} docs)")
        self._chat_log(f"retrieval_count={len(docs)}")
        
        if not docs:
            return self._build_chat_result("Xin lỗi, tôi không tìm thấy sản phẩm phù hợp với yêu cầu của bạn.", [], self._default_state_from_query(parsed_query, session_state), parsed_query)

        # 3. Re-ranking
        t1 = time.time()
        resolved_metas = [self._resolve_doc_metadata(d) for d in docs]
        candidate_indices = list(range(len(docs)))

        active_state = session_state if isinstance(session_state, ChatSessionState) else ChatSessionState.model_validate(session_state or {})
        # Only include active session state in the anchor when we have determined that
        # previous context is relevant. Including it unconditionally causes the re-ranker
        # to unfairly boost the previously active product even for unrelated queries.
        anchor_text = " ".join(
            part
            for part in [
                parsed_query.entities.get("product_name") if parsed_query.entities else None,
                parsed_query.entities.get("brand") if parsed_query.entities else None,
                parsed_query.entities.get("category") if parsed_query.entities else None,
                parsed_query.entities.get("color") if parsed_query.entities else None,
                parsed_query.entities.get("size") if parsed_query.entities else None,
                active_state.active_product_name if use_previous_context else None,
                active_state.active_brand if use_previous_context else None,
                active_state.active_category if use_previous_context else None,
            ]
            if part
        )

        # Keep reranker cost predictable on CPU.
        candidate_indices = candidate_indices[:8]
        candidate_docs = [docs[idx] for idx in candidate_indices]
        candidate_metas = [resolved_metas[idx] for idx in candidate_indices]
        doc_texts = [
            self._resolve_doc_context(d, m)
            for d, m in zip(candidate_docs, candidate_metas)
        ]
        pairs = [[search_query, text] for text in doc_texts]

        scores = self.reranker.predict(pairs)

        scored_items = []
        for local_idx, score in enumerate(scores):
            idx = candidate_indices[local_idx]
            meta = resolved_metas[idx]
            token_score = self._token_overlap_score(search_query, meta)
            anchor_score = max(
                self._score_doc_anchor(user_query, meta),
                self._score_doc_anchor(search_query, meta),
                self._score_doc_anchor(anchor_text, meta),
            )
            boosted_score = float(score) + (0.60 * token_score) + (0.90 * anchor_score)
            scored_items.append((idx, boosted_score, anchor_score, anchor_score >= 0.34))

        ranked_debug = []
        for idx, boosted_score, anchor_score, is_match in sorted(scored_items, key=lambda item: (item[1], item[2]), reverse=True)[:8]:
            ranked_debug.append({
                **self._compact_meta(resolved_metas[idx]),
                "score": round(float(boosted_score), 4),
                "anchor_score": round(float(anchor_score), 4),
                "match": is_match,
            })
        self._chat_log(f"ranked_candidates={ranked_debug}")

        scored_items.sort(key=lambda item: (item[1], item[2]), reverse=True)

        primary_index = self._select_primary_product_index(user_query, search_query, scored_items, resolved_metas)
        top_indices = [item[0] for item in scored_items]

        if self._has_explicit_product_signal(parsed_query):
            strict_anchor_indices = []
            for idx in top_indices:
                meta = resolved_metas[idx]
                explicit_anchor_score = self._score_doc_anchor(parsed_query.search_query, meta)
                if explicit_anchor_score >= 0.45:
                    strict_anchor_indices.append(idx)
            if strict_anchor_indices:
                top_indices = strict_anchor_indices
                if primary_index not in top_indices:
                    primary_index = top_indices[0]
                self._chat_log(
                    f"strict_anchor_filter applied | kept={[self._compact_meta(resolved_metas[idx]) for idx in top_indices[:5]]}"
                )

        if self._is_alternative_query(user_query):
            excluded_name = str((parsed_query.entities or {}).get("product_name") or "").strip()
            if excluded_name:
                excluded_idx = None
                excluded_score = 0.0
                for idx in top_indices:
                    score = self._score_doc_anchor(excluded_name, resolved_metas[idx])
                    if score > excluded_score:
                        excluded_score = score
                        excluded_idx = idx

                if excluded_idx is not None and excluded_score >= 0.6:
                    excluded_meta = resolved_metas[excluded_idx]
                    top_indices = [idx for idx in top_indices if idx != excluded_idx]
                    core_nouns = self._extract_core_product_nouns(excluded_name)
                    if core_nouns:
                        noun_filtered_indices = [
                            idx for idx in top_indices
                            if self._meta_matches_core_nouns(resolved_metas[idx], core_nouns)
                        ]
                        if noun_filtered_indices:
                            top_indices = noun_filtered_indices

                    self._chat_log(
                        f"alternative_query exclusion='{excluded_name}' excluded={self._compact_meta(excluded_meta)} score={excluded_score:.2f} remaining={[self._compact_meta(resolved_metas[idx]) for idx in top_indices[:5]]}"
                    )

        if parsed_query.intent in {"size_advice", "color_question"} and active_state.active_product_id:
            active_match = [
                idx for idx in top_indices
                if int(resolved_metas[idx].get("product_id", 0) or 0) == active_state.active_product_id
            ]
            if active_match:
                top_indices = active_match
                primary_index = active_match[0]
                self._chat_log(f"active_state_match product_id={active_state.active_product_id} -> kept_only_active_match")
            else:
                self._chat_log(
                    f"active_state_mismatch product_id={active_state.active_product_id} | top_candidates={[self._compact_meta(resolved_metas[idx]) for idx in top_indices[:3]]}"
                )
        elif primary_index is not None:
            top_indices = [primary_index] + [idx for idx in top_indices if idx != primary_index]
            self._chat_log(f"primary_index={primary_index} | selected={self._compact_meta(resolved_metas[primary_index])}")
        
        final_docs = []
        filtered_products = []
        
        for idx in top_indices[:3]:
            doc = docs[idx]
            final_docs.append(doc)
            meta = resolved_metas[idx]

            filtered_products.append(ProductInfo(
                id=int(meta.get("product_id", 0) or 0),
                name=meta.get("name", "Unknown"),
                price=float(meta.get("price", 0)),
                image_url=meta.get("primary_image_url") or meta.get("image_url", ""),
                category=meta.get("category_name") or meta.get("category", ""),
                brand=meta.get("brand_name") or meta.get("brand", ""),
                category_department=meta.get("category_department", "")
            ))
        
        print(f"⚖️ Re-ranking: {time.time() - t1:.2f}s. Final Docs: {len(final_docs)}")
        self._chat_log(f"final_docs={[self._compact_meta(self._resolve_doc_metadata(doc)) for doc in final_docs]}")

        # Fallback
        if not final_docs:
            print("⚠️ Fallback: Using top 2 docs from initial retrieval")
            for doc in docs[:2]:
                final_docs.append(doc)
                meta = self._resolve_doc_metadata(doc)
                filtered_products.append(ProductInfo(
                    id=int(meta.get("product_id", 0) or 0),
                    name=meta.get("name", "Unknown"),
                    price=float(meta.get("price", 0)),
                    image_url=meta.get("primary_image_url") or meta.get("image_url", ""),
                    category=meta.get("category_name") or meta.get("category", ""),
                    brand=meta.get("brand_name") or meta.get("brand", ""),
                    category_department=meta.get("category_department", "")
                ))

        if not final_docs:
            return self._build_chat_result("Xin lỗi, hiện tại shop chưa có đúng sản phẩm bạn đang tìm.", [], self._default_state_from_query(parsed_query, session_state), parsed_query)

        if parsed_query.intent in {"size_advice", "color_question"}:
            final_docs = final_docs[:1]
            filtered_products = filtered_products[:1]

        # Deterministic answer for size questions when metadata already has size fields.
        size_answer = self._build_size_answer(user_query, final_docs)
        if size_answer:
            state = self._build_state_from_meta(parsed_query, self._resolve_doc_metadata(final_docs[0]), session_state)
            self._chat_log(f"branch=size_answer | next_state={state.model_dump()}")
            return self._build_chat_result(size_answer, filtered_products, state, parsed_query)

        color_answer = self._build_color_answer(user_query, final_docs)
        if color_answer:
            state = self._build_state_from_meta(parsed_query, self._resolve_doc_metadata(final_docs[0]), session_state)
            self._chat_log(f"branch=color_answer | next_state={state.model_dump()}")
            return self._build_chat_result(color_answer, filtered_products, state, parsed_query)

        # 4. Generation
        t2 = time.time()
        context_text = ""
        for doc in final_docs:
            meta = self._resolve_doc_metadata(doc)
            doc_context = self._resolve_doc_context(doc, meta)
            context_text += (
                f"- Product: {meta.get('name')}\n"
                f"  Price: ${meta.get('price')}\n"
                f"  Details: {doc_context}\n\n"
            )

        chain = self.product_qa_prompt | self.llm_main | StrOutputParser()
        try:
            answer = chain.invoke({
                "context": context_text,
                "question": user_query 
            })
            answer = answer.replace("Answer:", "").strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            answer = "Hệ thống đang bận, vui lòng thử lại sau ít phút."

        print(f"🤖 Generation: {time.time() - t2:.2f}s")
        state = self._build_state_from_meta(parsed_query, self._resolve_doc_metadata(final_docs[0]), session_state)
        self._chat_log(f"branch=generation | next_state={state.model_dump()}")
        return self._build_chat_result(answer, filtered_products, state, parsed_query)
    
    def upsert_product(self, data: ProductSyncRequest):
        print(f"🔄 Syncing product: {data.name} (ID: {data.id})")

        product_brand = data.brand_name or data.brand or "Generic"
        product_category = data.category_name or data.category or "General"
        product_image_url = data.primary_image_url or data.image_url or ""
        product_department = data.category_department or ""

        variants_text = ""
        size_names = set()
        color_names = set()
        if data.variants:
            variant_chunks = []
            for variant in data.variants:
                color = variant.get("color_name") or "Unknown color"
                size = variant.get("size_name") or "Unknown size"
                color_names.add(str(color).strip())
                size_names.add(str(size).strip())
                stock = int(variant.get("stock_quantity") or 0)
                sold = int(variant.get("sold_quantity") or 0)
                sku = variant.get("sku") or ""
                sku_part = f" | SKU: {sku}" if sku else ""
                variant_chunks.append(f"{color} / {size}{sku_part} | stock={stock} | sold={sold}")
            variants_text = "; ".join(variant_chunks)

        try:
            self.client.delete(
                collection_name=COLLECTION_PRODUCT_TEXT,
                points_selector=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="product_id",
                            match=models.MatchValue(value=data.id)
                        ),
                        models.FieldCondition(
                            key="metadata.product_id", 
                            match=models.MatchValue(value=data.id)
                        )
                    ]
                )
            )
        except Exception as e:
            print(f"⚠️ Clean up old vector failed: {e}")

        full_text = (
            f"Product: {data.name}. "
            f"Brand: {product_brand}. "
            f"Category: {product_category}. "
            f"Department: {product_department}. "
            f"Details: {data.description}. "
            f"Features: {data.attributes}. "
            f"Variants: {variants_text}. "
            f"Price: {data.price}."
        )

        meta = {
            "product_id": data.id,
            "name": data.name,
            "price": data.price,
            "image_url": product_image_url,
            "primary_image_url": product_image_url,
            "category": product_category,
            "category_name": product_category,
            "category_department": product_department,
            "brand": product_brand,
            "brand_name": product_brand,
            "size_names": sorted([s for s in size_names if s]),
            "color_names": sorted([c for c in color_names if c]),
            "variants": data.variants or []
        }

        # [FIX] Đổi tên biến cho đúng
        doc = Document(page_content=full_text, metadata=meta)
        self.product_store.add_documents([doc])
        print(f"Synced successfully: ID {data.id}")
        return True

    def delete_product(self, product_id: int):
        print(f"Deleting product ID: {product_id}")
        self.client.delete(
            collection_name=COLLECTION_PRODUCT_TEXT,
            points_selector=models.Filter(
                should=[
                    models.FieldCondition(
                        key="product_id", 
                        match=models.MatchValue(value=product_id)
                    ),
                    models.FieldCondition(
                        key="metadata.product_id", 
                        match=models.MatchValue(value=product_id)
                    )
                ]
            )
        )

        # Keep image collection in sync for remove operations.
        self.client.delete(
            collection_name=COLLECTION_PRODUCT_IMAGE,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="product_id",
                        match=models.MatchValue(value=product_id),
                    )
                ]
            ),
        )
        return True
    
    def moderate_content(self, text: str):
        """
        Dùng LLM (Llama 3) để phân tích sâu ngữ nghĩa (Sarcasm, Hate Speech).
        Chỉ gọi hàm này khi Local Model cảm thấy không chắc chắn.
        """
        # Prompt chuyên dụng cho Fashion E-commerce
        prompt = ChatPromptTemplate.from_template("""
        You are a strict Content Moderator for a Fashion E-commerce site.
        
        Task: Analyze the user review: "{text}"
        
        Determine if it is TOXIC based on these rules:
        1. **TOXIC (REJECT)**: 
           - Contains profanity (e.g., f*ck, sh*t).
           - Personal attacks against staff or other users (e.g., "you are stupid").
           - Hate speech, racism, sexism.
           - Spam or scam links.
           
        2. **SAFE (APPROVE)**:
           - Negative reviews about the PRODUCT (e.g., "The fabric is ugly", "Wrong size").
           - Negative reviews about SERVICE (e.g., "Shipping was slow", "Support didn't reply").
           - Sarcasm that is NOT insulting (e.g., "Great, arrived 2 weeks late").

        Output valid JSON only:
        {{
            "is_toxic": boolean,
            "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
            "reason": "short explanation"
        }}
        """)
        
        # Dung model llm_fast cho nhanh va tiet kiem chi phi
        chain = prompt | self.llm_fast | StrOutputParser()
        
        try:
            result_str = chain.invoke({"text": text})
            
            # Clean JSON (Phòng trường hợp LLM trả về dư text)
            import json
            start = result_str.find('{')
            end = result_str.rfind('}') + 1
            json_str = result_str[start:end]
            return json.loads(json_str)
            
        except Exception as e:
            print(f"⚠️ LLM Moderation Error: {e}")
            # Fallback an toàn: Coi như không Toxic nhưng là Neutral
            return {"is_toxic": False, "sentiment": "NEUTRAL", "reason": "Error parsing LLM"}

    def moderate_english_content(self, text: str):
        """Backward compatibility wrapper."""
        return self.moderate_content(text)