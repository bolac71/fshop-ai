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
    PROJECT_ROOT,
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

RERANK_TOP_K = int(os.getenv("AI_RERANK_TOP_K", "4"))
ENABLE_RERANK = os.getenv("AI_ENABLE_RERANK", "true").strip().lower() in {"1", "true", "yes", "on"}

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
        "tee", "thun", "polo", "jacket", "sweater", "short", "skirt", "dirtycoins", "converse",
        "nike", "adidas", "mlb", "new", "balance", "core", "vans",
    }

    DISCOVERY_KEYWORDS = {
        "co", "ban", "khong", "shop", "tim", "goi", "y", "mua", "muon",
        "recommend", "suggest", "sell", "have",
    }

    CORE_PRODUCT_NOUNS = {
        "quan", "ao", "thun", "vay", "dam", "giay", "dep", "hoodie", "jacket", "khoac", "balo",
        "shirt", "tee", "polo", "short", "skirt", "jean", "jeans",
    }

    def __init__(self, client):
        self.client = client
        self.query_intent_service = QueryIntentService()
        self.chat_debug_enabled = os.getenv("AI_CHAT_DEBUG", "true").strip().lower() in {"1", "true", "yes", "on"}
        self._catalog_index: dict | None = None
        self._catalog_index_loaded_at = 0.0
        self._style_profiles: dict | None = None

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
        Bạn là nhân viên chăm sóc khách hàng thân thiện của cửa hàng thời trang FShop.
        
        DỮ LIỆU ĐƠN HÀNG CỦA KHÁCH HÀNG:
        {order_context}
        
        CÂU HỎI CỦA KHÁCH HÀNG: {question}
        
        HƯỚNG DẪN TRẢ LỜI:
        1. Chỉ trả lời dựa vào dữ liệu đơn hàng được cung cấp ở trên.
        2. Hãy tóm tắt trạng thái các đơn hàng một cách rõ ràng và thân thiện (Ví dụ: đơn hàng nào đang xử lý, đơn nào đang giao, sản phẩm là gì, tổng tiền bao nhiêu).
        3. Trạng thái 'Chờ xác nhận' (pending), 'Đã xác nhận' (confirmed), 'Đang xử lý' (processing): Cho biết đơn hàng đang được chuẩn bị và đóng gói.
        4. Trạng thái 'Đang giao hàng' / 'Đang vận chuyển' (in_transit/out_for_delivery/shipped): Cho biết đơn hàng đang trên đường vận chuyển.
        5. Trạng thái 'Đã giao thành công' (delivered): Chúc khách hàng sử dụng sản phẩm vui vẻ.
        6. Trạng thái 'Đã hủy' (canceled): Thông báo đơn đã hủy.
        7. Nếu khách hàng hỏi về cách hủy đơn, hãy hướng dẫn họ vào ứng dụng/website để thực hiện yêu cầu hủy (thường chỉ hủy được khi đơn hàng ở trạng thái Chuẩn bị/Chờ xác nhận/Đóng gói).
        8. Nếu dữ liệu đơn hàng trống hoặc không có đơn nào phù hợp, hãy thông báo lịch sự rằng không tìm thấy đơn hàng tương ứng và nhắc họ kiểm tra lại mã đơn hàng (Order ID).
        9. Luôn trả lời bằng tiếng Việt có dấu lịch sự, chân thành.
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

    def _normalize_parser_payload(self, payload: dict, question: str) -> dict:
        if not isinstance(payload, dict):
            return {}

        payload["search_query"] = str(payload.get("search_query") or question).strip()
        payload["intent"] = self._normalize_parser_intent(str(payload.get("intent") or "unknown"))
        payload["requires_context"] = bool(payload.get("requires_context", False))
        payload["follow_up"] = bool(payload.get("follow_up", False))
        payload["entities"] = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}

        try:
            payload["confidence"] = float(payload.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            payload["confidence"] = 0.0

        return payload

    def _validate_parsed_query(self, parsed: ParsedChatQuery, question: str) -> ParsedChatQuery | None:
        if not parsed.search_query or not parsed.search_query.strip():
            parsed.search_query = question

        parsed.intent = self._normalize_parser_intent(parsed.intent)
        if parsed.intent == "size_advice" and self._is_size_support_policy_question(question):
            parsed.intent = "policy"

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
                payload = self._normalize_parser_payload(json.loads(self._extract_json_payload(raw)), question)
                parsed = ParsedChatQuery.model_validate(payload)
                parsed = self._validate_parsed_query(parsed, question)
                parsed = self._apply_deterministic_intent_guard(parsed, question)
                if parsed and parsed.confidence >= 0.35:
                    return parsed
            except Exception as e:
                print(f"Parser fallback error: {e}")

        return self._default_parsed_query(question)

    def _apply_deterministic_intent_guard(self, parsed: ParsedChatQuery, question: str) -> ParsedChatQuery:
        fast_intent = self._detect_intent_fast(question).lower()
        if fast_intent in {"order", "policy"} and parsed.intent not in {"order", "policy"}:
            parsed.intent = fast_intent
            parsed.search_query = question
            parsed.requires_context = False
            parsed.follow_up = False
            parsed.confidence = max(parsed.confidence, 0.85)
        return parsed

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

            status_map = {
                'pending': 'Chờ xác nhận',
                'confirmed': 'Đã xác nhận',
                'awaiting_pickup': 'Chờ lấy hàng',
                'in_transit': 'Đang vận chuyển',
                'out_for_delivery': 'Đang giao hàng',
                'shipped': 'Đang vận chuyển',
                'delivered': 'Đã giao thành công',
                'delivery_failed': 'Giao hàng thất bại',
                'canceled': 'Đã hủy',
                'cancelled': 'Đã hủy',
                'processing': 'Đang xử lý',
            }

            order_text = ""
            for row in rows:
                oid, status, total, created_at, items = row
                
                status_vn = status_map.get(str(status).lower(), status)
                
                try:
                    total_val = float(total)
                    total_formatted = "{:,.0f} VND".format(total_val).replace(",", ".")
                except Exception:
                    total_formatted = f"{total} VND"
                    
                if hasattr(created_at, "strftime"):
                    date_formatted = created_at.strftime("%d/%m/%Y %H:%M")
                else:
                    date_formatted = str(created_at)
                    
                order_text += (
                    f"- Mã đơn hàng: #{oid}\n"
                    f"  Trạng thái: {status_vn}\n"
                    f"  Tổng tiền: {total_formatted}\n"
                    f"  Ngày mua: {date_formatted}\n"
                    f"  Sản phẩm: {items}\n\n"
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

        # Nếu có từ khóa chính sách kết hợp với từ khóa nghiệp vụ khác, ưu tiên POLICY
        policy_indicators = ["chinh sach", "quy dinh", "huong dan", "dieu kien", "cach thuc"]
        is_policy_indicated = any(pi in q for pi in policy_indicators)

        order_keywords = [
            "order", "track", "package", "delivery", "status", "shipment", "cancel", "bought", "purchase history",
            "don hang", "ma don", "theo doi", "van don", "trang thai don", "don cua toi",
            "don minh", "don cua minh", "kiem tra don", "huy don", "lich su mua", "da mua",
        ]
        
        policy_keywords = [
            "policy", "shipping", "ship", "delivery", "deliver",
            "return", "refund", "exchange", "money back",
            "payment", "pay", "credit card", "cod", "cash",
            "address", "location", "store", "shop open", "hour", "contact", "phone",
            "chinh sach", "phi ship", "van chuyen", "doi tra", "hoan tien", "thanh toan",
            "dia chi", "cua hang", "gio mo cua", "lien he", "so dien thoai", "kich co", "bang size"
        ]

        if is_policy_indicated:
            return "POLICY"

        # Nếu hỏi chung chung dạng "làm thế nào để...", "như thế nào..." mà không đi kèm mã số đơn hàng cụ thể
        how_indicators = ["the nao", "ra sao", "lam sao", "lam the nao"]
        has_number = bool(re.search(r'\d+', q))
        if any(hi in q for hi in how_indicators) and not has_number:
            if any(kw in q for kw in ["huy", "doi tra", "ship", "giao", "tra", "hoan tien", "thanh toan"]):
                return "POLICY"

        has_order_word = any(term in q for term in ["don hang", "ma don", "don cua", "don minh", "order", "van don"])
        if has_order_word or any(kw in q for kw in order_keywords):
            return "ORDER"
        
        if any(kw in q for kw in policy_keywords):
            return "POLICY"
        
        return "PRODUCT"

    def _is_policy_overview_question(self, question: str) -> bool:
        q = self._normalize_text(question)
        overview_patterns = [
            r"\b(co|gom|nhung|cac)\b.*\b(chinh sach|policy)\b",
            r"\b(chinh sach|policy)\b.*\b(gi|nao|nhung gi|cac loai|bao gom)\b",
            r"\bshop\b.*\b(chinh sach|policy)\b",
        ]
        return any(re.search(pattern, q) for pattern in overview_patterns)

    def _is_size_support_policy_question(self, question: str) -> bool:
        q = self._normalize_text(question)
        support_signal = any(phrase in q for phrase in [
            "tu van size",
            "ho tro size",
            "ho tro tu van size",
            "bang size",
            "huong dan chon size",
            "chinh sach size",
        ])
        if not support_signal:
            return False

        product_signal = self._is_explicit_product_query(question)
        body_metric_signal = bool(re.search(r"\b\d{2,3}\s*kg\b|\b\d{3}\s*cm\b|\b1m\d{2}\b", q))
        return not product_signal and not body_metric_signal

    def _get_all_policy_docs(self) -> list[Document]:
        try:
            points, _ = self.client.scroll(
                collection_name=COLLECTION_POLICIES,
                limit=64,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            print(f"Policy scroll failed, fallback to vector search: {exc}")
            return []

        docs = []
        for point in points:
            payload = point.payload or {}
            content = payload.get("page_content") or payload.get("text") or ""
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            if content:
                docs.append(Document(page_content=content, metadata=metadata))

        return sorted(docs, key=lambda doc: int((doc.metadata or {}).get("chunk", 999)))

    def _build_policy_overview_answer(self, docs: list[Document]) -> str:
        titles = []
        for doc in docs:
            for line in (doc.page_content or "").splitlines():
                line = line.strip()
                if re.match(r"^\d+\.\s+", line):
                    titles.append(line)
                    break

        if not titles:
            return "FShop hiện có các chính sách về vận chuyển, thanh toán, đổi trả, hủy đơn, tư vấn size, bảo mật thông tin và khuyến mãi."

        lines = "\n".join(f"- {title}" for title in titles)
        return f"Chào bạn, FShop hiện có các chính sách sau:\n\n{lines}\n\nBạn muốn mình giải thích chi tiết chính sách nào?"
    
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

    def _catalog_index_stale(self) -> bool:
        ttl_seconds = int(os.getenv("AI_CATALOG_INDEX_TTL_SECONDS", "300"))
        return not self._catalog_index or (time.time() - self._catalog_index_loaded_at) > ttl_seconds

    def _get_catalog_index(self) -> dict:
        if not self._catalog_index_stale():
            return self._catalog_index or {}

        index = {
            "products": [],
            "categories": {},
            "brands": {},
            "colors": {},
            "sizes": {},
        }
        try:
            points, _ = self.client.scroll(
                collection_name=COLLECTION_PRODUCT_TEXT,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            print(f"Catalog index build failed: {exc}")
            return self._catalog_index or index

        for point in points:
            payload = point.payload or {}
            if not payload.get("product_id"):
                continue
            index["products"].append(payload)
            self._add_taxonomy_value(
                index["categories"],
                payload.get("category_name"),
                payload.get("category_slug"),
                extra_aliases=self._category_general_aliases(payload.get("category_name")),
            )
            self._add_taxonomy_value(index["brands"], payload.get("brand_name"), payload.get("brand_slug"))
            for color in payload.get("color_names") or []:
                self._add_taxonomy_value(index["colors"], color)
            for size in payload.get("size_names") or []:
                self._add_taxonomy_value(index["sizes"], size)

        self._catalog_index = index
        self._catalog_index_loaded_at = time.time()
        self._chat_log(
            f"catalog_index loaded products={len(index['products'])} categories={len(index['categories'])} brands={len(index['brands'])}"
        )
        return index

    def _category_general_aliases(self, category_name: str | None) -> set[str]:
        normalized = self._normalize_text(str(category_name or ""))
        if not normalized:
            return set()
        aliases = set()
        generalized = re.sub(r"\b(nam|nu|tre em|unisex|be trai|be gai)\b", " ", normalized)
        generalized = re.sub(r"\s+", " ", generalized).strip()
        if generalized and generalized != normalized:
            aliases.add(generalized)
        return aliases

    def _add_taxonomy_value(self, bucket: dict, value: str | None, slug: str | None = None, extra_aliases: set[str] | None = None):
        if not value:
            return
        display = str(value).strip()
        aliases = {
            self._normalize_text(display),
            self._normalize_text(slug or ""),
            self._normalize_text(display.replace("-", " ")),
        }
        aliases.update(extra_aliases or set())
        aliases = {alias for alias in aliases if alias}
        primary = self._normalize_text(display)
        if not primary:
            return
        current = bucket.setdefault(primary, {"display": display, "aliases": set()})
        current["aliases"].update(aliases)

    def _find_taxonomy_match(self, query: str, bucket: dict) -> dict | None:
        matches = self._find_taxonomy_matches(query, bucket)
        return matches[0] if matches else None

    def _find_taxonomy_matches(self, query: str, bucket: dict) -> list[dict]:
        q = self._normalize_text(query)
        matches = []
        for key, item in bucket.items():
            best_alias = ""
            for alias in item.get("aliases", set()):
                if not alias:
                    continue
                pattern = rf"(^|\s){re.escape(alias)}(\s|$)"
                if re.search(pattern, q) and len(alias) > len(best_alias):
                    best_alias = alias
            if best_alias:
                matches.append({"key": key, "display": item.get("display", key), "alias": best_alias})
        matches.sort(key=lambda item: len(item.get("alias", "")), reverse=True)
        return matches

    def _extract_structured_filters(self, query: str) -> dict:
        index = self._get_catalog_index()
        filters = {}
        category_matches = self._find_taxonomy_matches(query, index.get("categories", {}))
        if category_matches:
            longest_alias_len = len(category_matches[0].get("alias", ""))
            filters["category"] = [
                match for match in category_matches
                if len(match.get("alias", "")) == longest_alias_len
            ]

        for key, bucket_name in [
            ("brand", "brands"),
            ("color", "colors"),
            ("size", "sizes"),
        ]:
            match = self._find_taxonomy_match(query, index.get(bucket_name, {}))
            if match:
                filters[key] = match

        q = self._normalize_text(query)
        if re.search(r"(^|\s)(nam|men|male)(\s|$)", q):
            filters["department"] = {"key": "men", "display": "Nam", "alias": "nam"}
        elif re.search(r"(^|\s)(nu|women|female)(\s|$)", q):
            filters["department"] = {"key": "women", "display": "Nữ", "alias": "nu"}
        elif re.search(r"(^|\s)(tre em|kids|kid|be trai|be gai)(\s|$)", q):
            filters["department"] = {"key": "kids", "display": "Trẻ em", "alias": "tre em"}

        return filters

    def _has_structured_filter(self, query: str) -> bool:
        filters = self._extract_structured_filters(query)
        return any(key in filters for key in ("category", "brand", "color", "size", "department"))

    def _has_hard_structured_filters(self, filters: dict) -> bool:
        return any(key in filters for key in ("category", "brand", "color", "size", "department"))

    def _meta_passes_structured_filters(self, meta: dict, filters: dict) -> bool:
        checks = [
            ("category", meta.get("category_name") or meta.get("category")),
            ("brand", meta.get("brand_name") or meta.get("brand")),
            ("color", " ".join(meta.get("color_names") or [])),
            ("size", " ".join(meta.get("size_names") or [])),
            ("department", meta.get("category_department")),
        ]
        for key, value in checks:
            expected = filters.get(key)
            if not expected:
                continue
            actual = self._normalize_text(str(value or ""))
            expected_items = expected if isinstance(expected, list) else [expected]
            expected_keys = [item.get("key") for item in expected_items if item.get("key")]
            if expected_keys and not any(expected_key in actual for expected_key in expected_keys):
                return False
        return True

    def _structured_filter_score(self, meta: dict, filters: dict) -> float:
        if not filters:
            return 0.0
        checks = [
            ("category", meta.get("category_name") or meta.get("category")),
            ("brand", meta.get("brand_name") or meta.get("brand")),
            ("color", " ".join(meta.get("color_names") or [])),
            ("size", " ".join(meta.get("size_names") or [])),
            ("department", meta.get("category_department")),
        ]
        hits = 0
        total = 0
        for key, value in checks:
            expected = filters.get(key)
            if not expected:
                continue
            total += 1
            actual = self._normalize_text(str(value or ""))
            expected_items = expected if isinstance(expected, list) else [expected]
            if any(item.get("key") in actual for item in expected_items if item.get("key")):
                hits += 1
        return hits / max(total, 1)

    def _get_structured_product_docs(self, query: str, limit: int = 24) -> tuple[list[Document], dict]:
        filters = self._extract_structured_filters(query)
        style_preference = self._extract_style_preference(query)
        if style_preference:
            filters = {**filters, "style": style_preference}

        if not filters:
            return [], filters

        docs = []
        scored_payloads = []
        index = self._get_catalog_index()
        has_hard_filters = self._has_hard_structured_filters(filters)
        for payload in index.get("products", []):
            if has_hard_filters and not self._meta_passes_structured_filters(payload, filters):
                continue
            token_score = self._token_overlap_score(query, payload)
            filter_score = self._structured_filter_score(payload, filters)
            style_score = self._style_color_score(payload, style_preference)
            if not has_hard_filters and style_score <= 0:
                continue
            stock_score = 0.05 if any(
                isinstance(variant, dict) and int(variant.get("stock_quantity") or 0) > 0
                for variant in (payload.get("variant_summary") or payload.get("variants") or [])
            ) else 0.0
            scored_payloads.append((filter_score + (0.85 * style_score) + token_score + stock_score, payload))

        scored_payloads.sort(key=lambda item: item[0], reverse=True)
        for _, payload in scored_payloads[:limit]:
            content = payload.get("search_text") or self._resolve_doc_context(Document(page_content="", metadata=payload), payload)
            docs.append(Document(page_content=content, metadata=payload))
        return docs, filters

    def _merge_unique_docs(self, preferred_docs: list[Document], fallback_docs: list[Document]) -> list[Document]:
        merged = []
        seen_ids = set()

        for doc in [*preferred_docs, *fallback_docs]:
            meta = self._resolve_doc_metadata(doc)
            product_id = meta.get("product_id") or meta.get("id") or (doc.metadata or {}).get("_id")
            if product_id in seen_ids:
                continue
            seen_ids.add(product_id)
            merged.append(doc)

        return merged

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
        if self._has_structured_filter(parsed_query.search_query or ""):
            return False
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

    def _is_recommendation_question(self, question: str) -> bool:
        q = self._normalize_text(question)
        return any(
            keyword in q
            for keyword in [
                "goi y", "de xuat", "nen chon", "chon gi", "chon mau gi",
                "chon size nao", "hop voi", "phu hop", "recommend", "suggest",
            ]
        )

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

    def _load_style_profiles(self) -> dict:
        cached = getattr(self, "_style_profiles", None)
        if cached is not None:
            return cached

        profile_path = PROJECT_ROOT / "app" / "data" / "style_profiles.json"
        try:
            with profile_path.open("r", encoding="utf-8") as file:
                raw_profiles = json.load(file)
        except Exception as exc:
            self._chat_log(f"style_profile_load_failed path={profile_path} error={exc}")
            raw_profiles = {}

        profiles = {}
        for key, profile in raw_profiles.items():
            if not isinstance(profile, dict):
                continue
            keywords = [
                self._normalize_text(str(keyword))
                for keyword in profile.get("keywords", [])
                if str(keyword).strip()
            ]
            preferred_colors = [
                self._normalize_text(str(color))
                for color in profile.get("preferred_colors", [])
                if str(color).strip()
            ]
            profiles[key] = {
                **profile,
                "key": key,
                "keywords_norm": keywords,
                "preferred_colors_norm": preferred_colors,
            }

        self._style_profiles = profiles
        return profiles

    def _extract_style_preference(self, query: str) -> dict | None:
        q = self._normalize_text(query or "")
        if not q:
            return None

        best_profile = None
        best_score = 0
        for profile in self._load_style_profiles().values():
            score = 0
            for keyword in profile.get("keywords_norm", []):
                if not keyword:
                    continue
                if re.search(rf"(^|\s){re.escape(keyword)}(\s|$)", q) or keyword in q:
                    score += max(1, len(keyword.split()))
            if score > best_score:
                best_score = score
                best_profile = profile

        return best_profile if best_score > 0 else None

    def _style_context_query(self, query: str, history: list | None) -> str:
        if self._extract_style_preference(query):
            return query

        history_parts = []
        for item in (history or [])[-6:]:
            if isinstance(item, dict):
                text = item.get("content") or item.get("message") or item.get("text") or ""
            else:
                text = getattr(item, "content", None) or str(item)
            if text and self._extract_style_preference(text):
                history_parts.append(str(text))

        return f"{' '.join(history_parts)} {query}".strip() if history_parts else query

    def _style_color_score(self, meta: dict, style_preference: dict | None) -> float:
        if not style_preference:
            return 0.0

        product_colors = [self._normalize_text(color) for color in self._extract_color_names(meta)]
        preferred_colors = style_preference.get("preferred_colors_norm") or []
        if not product_colors or not preferred_colors:
            return 0.0

        hits = 0
        for color in product_colors:
            if any(preferred in color or color in preferred for preferred in preferred_colors):
                hits += 1

        return min(1.0, hits / max(len(product_colors), 1))

    def _recommend_color_for_style(self, color_names: list[str], style_preference: dict | None) -> str | None:
        if not color_names or not style_preference:
            return None

        preferred_colors = style_preference.get("preferred_colors_norm") or []
        for preferred in preferred_colors:
            for color in color_names:
                normalized_color = self._normalize_text(color)
                if preferred in normalized_color or normalized_color in preferred:
                    return color

        return None

    def _build_style_color_sentence(self, question: str, meta: dict) -> str | None:
        style_preference = self._extract_style_preference(question)
        color_names = self._extract_color_names(meta)
        recommended_color = self._recommend_color_for_style(color_names, style_preference)
        if not style_preference or not recommended_color:
            return None

        display = style_preference.get("display") or "phong cách này"
        reason = style_preference.get("reason") or "hợp với phong cách bạn đang tìm"
        return (
            f"Về màu, mình nghiêng về {recommended_color} cho vibe {display}, "
            f"vì {reason}."
        )

    def _build_combined_size_color_answer(self, question: str, final_docs: list[Document]) -> str | None:
        if not final_docs:
            return None

        # Check if user asks about both size and color
        is_size = self._is_size_question(question)
        is_color = self._is_color_question(question)
        
        if not (is_size and is_color):
            return None

        doc = final_docs[0]
        meta = self._resolve_doc_metadata(doc)
        size_names = self._extract_size_names(meta)
        color_names = self._extract_color_names(meta)
        
        if not size_names and not color_names:
            return None

        product_name = meta.get("name", "sản phẩm")
        
        parts = []
        if size_names:
            sizes_text = ", ".join(size_names)
            parts.append(f"size: {sizes_text}")
        if color_names:
            colors_text = ", ".join(color_names)
            parts.append(f"màu: {colors_text}")

        intro = f"{product_name} hiện có các " + " và ".join(parts) + "."
        
        # Recommendations
        rec_parts = []
        
        # Size recommendation based on body metrics
        weight_kg, height_cm = self._extract_body_metrics(question)
        if (weight_kg or height_cm) and size_names:
            recommended_size = self._recommend_size_from_metrics(size_names, weight_kg, height_cm)
            metrics_parts = []
            if weight_kg:
                metrics_parts.append(f"{weight_kg}kg")
            if height_cm:
                metrics_parts.append(f"{height_cm}cm")
            metrics_text = " / ".join(metrics_parts)
            rec_parts.append(f"Với thông tin {metrics_text}, mình đề xuất bạn thử size {recommended_size} (nếu thích rộng rãi thì có thể tăng lên 1 size)")

        # Color recommendation based on style preference
        style_preference = self._extract_style_preference(question)
        recommended_color = self._recommend_color_for_style(color_names, style_preference)
        if style_preference and recommended_color:
            display = style_preference.get("display") or "phong cách này"
            reason = style_preference.get("reason") or "hợp với vibe bạn đang tìm"
            rec_parts.append(f"về màu sắc, để phù hợp với vibe {display}, mình đề xuất chọn màu {recommended_color} vì {reason}")

        if rec_parts:
            rec_text = ". ".join(rec_parts) + "."
            answer = f"{intro} {rec_text}"
        else:
            prompt_tips = []
            if size_names:
                prompt_tips.append("chiều cao/cân nặng để mình tư vấn size")
            if color_names:
                prompt_tips.append("phong cách để mình gợi ý màu sắc")
            tips_text = " hoặc ".join(prompt_tips)
            answer = f"{intro} Bạn cần mình gợi ý cụ thể hơn theo {tips_text} không?"

        return answer

    def _build_preference_followup_answer(self, question: str, final_docs: list[Document]) -> str | None:
        if not final_docs:
            return None

        style_preference = self._extract_style_preference(question)
        weight_kg, height_cm = self._extract_body_metrics(question)
        asks_for_suggestion = self._is_recommendation_question(question)
        if not style_preference and not weight_kg and not height_cm and not asks_for_suggestion:
            return None

        meta = self._resolve_doc_metadata(final_docs[0])
        size_names = self._extract_size_names(meta)
        color_names = self._extract_color_names(meta)
        if not size_names and not color_names:
            return None

        product_name = meta.get("name", "sản phẩm")
        parts = []

        if color_names:
            colors_text = ", ".join(color_names)
            recommended_color = self._recommend_color_for_style(color_names, style_preference)
            if recommended_color and style_preference:
                display = style_preference.get("display") or "phong cách này"
                reason = style_preference.get("reason") or "hợp với vibe bạn đang tìm"
                parts.append(
                    f"Về màu, {product_name} có {colors_text}; mình đề xuất {recommended_color} cho vibe {display}, vì {reason}"
                )
            elif asks_for_suggestion:
                parts.append(
                    f"Về màu, {product_name} hiện có {colors_text}; bạn có thể chọn màu tối hoặc trung tính nếu muốn dễ phối hơn"
                )

        if size_names:
            sizes_text = ", ".join(size_names)
            if weight_kg or height_cm:
                recommended_size = self._recommend_size_from_metrics(size_names, weight_kg, height_cm)
                metrics_parts = []
                if weight_kg:
                    metrics_parts.append(f"{weight_kg}kg")
                if height_cm:
                    metrics_parts.append(f"{height_cm}cm")
                metrics_text = " / ".join(metrics_parts)
                parts.append(
                    f"Về size, sản phẩm có {sizes_text}; với thông tin {metrics_text}, mình nghĩ bạn nên thử size {recommended_size}"
                )
            elif asks_for_suggestion:
                parts.append(
                    f"Về size, sản phẩm có {sizes_text}; gửi mình chiều cao/cân nặng nếu bạn muốn mình chốt size kỹ hơn"
                )

        if not parts:
            return None

        return ". ".join(parts) + "."

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
                style_color_sentence = self._build_style_color_sentence(question, meta)

                answer = (
                    f"{product_name} hiện có các size: {sizes_text}. "
                    f"Với thông tin {metrics_text}, mình đề xuất bạn thử size {recommended_size}. "
                    "Nếu bạn thích mặc rộng hơn, có thể tăng lên 1 size."
                )
                if style_color_sentence:
                    answer = f"{answer} {style_color_sentence}"
                return answer

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
            style_preference = self._extract_style_preference(question)
            recommended_color = self._recommend_color_for_style(color_names, style_preference)
            if recommended_color and style_preference:
                display = style_preference.get("display") or "phong cách này"
                reason = style_preference.get("reason") or "hợp với vibe bạn đang tìm"
                return (
                    f"{product_name} hiện có các màu: {colors_text}. "
                    f"Nếu bạn muốn vibe {display}, mình đề xuất chọn màu {recommended_color}, "
                    f"vì {reason}."
                )

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
        text = text.replace("đ", "d")
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
        if intent == "order":
            if not user_id:
                state = self._default_state_from_query(parsed_query, session_state)
                return self._build_chat_result("Vui lòng đăng nhập để kiểm tra đơn hàng.", [], state, parsed_query)
            
            # Trích xuất mã đơn hàng nếu có (Ví dụ: "Đơn hàng 123", "Mã đơn 123")
            normalized_query = self._normalize_text(user_query)
            order_id_match = re.search(r'(?:order|don hang|don|ma don|ma|#)\s*(\d+)', normalized_query)
            specific_id = order_id_match.group(1) if order_id_match else None
            
            # Lấy data SQL
            order_data = self._get_user_orders_sql(user_id, specific_id)
            
            # Sinh câu trả lời
            chain = self.order_qa_prompt | self.llm_main | StrOutputParser()
            answer = chain.invoke({"order_context": order_data, "question": user_query})
            
            print(f"⏱️ TOTAL TIME: {time.time() - total_start:.2f}s")
            state = self._default_state_from_query(parsed_query, session_state)
            return self._build_chat_result(answer, [], state, parsed_query)

        # CASE B: POLICY
        elif intent == "policy":
            t0 = time.time()
            if self._is_policy_overview_question(user_query):
                docs = self._get_all_policy_docs()
                retrieval_mode = "all"
            else:
                docs = self.policy_store.similarity_search(user_query, k=4)
                retrieval_mode = "similarity"

            print(f"🔍 Policy Retrieval ({retrieval_mode}): {time.time() - t0:.2f}s")
            
            state = self._default_state_from_query(parsed_query, session_state)
            if not docs:
                return self._build_chat_result(
                    "Xin lỗi, hiện tại mình chưa tìm thấy thông tin chính sách phù hợp. Bạn có thể hỏi cụ thể về vận chuyển, đổi trả, thanh toán hoặc liên hệ shop nhé.",
                    [],
                    state,
                    parsed_query,
                )

            if self._is_policy_overview_question(user_query):
                answer = self._build_policy_overview_answer(docs)
                return self._build_chat_result(answer, [], state, parsed_query)
            
            context_text = "\n\n".join([d.page_content for d in docs])
            chain = self.policy_qa_prompt | self.llm_main | StrOutputParser()
            answer = chain.invoke({"context": context_text, "question": user_query})
            return self._build_chat_result(answer, [], state, parsed_query)

        # CASE C: PRODUCT (Default)
        else:
            return await self._handle_product_search(user_query, history, parsed_query, session_state)

    async def _handle_product_search(self, user_query: str, history: list, parsed_query: ParsedChatQuery, session_state: ChatSessionState | dict | None):
        search_query = parsed_query.search_query or user_query
        use_previous_context = self._should_use_previous_context(parsed_query) and not self._has_structured_filter(user_query)
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
        style_context_query = self._style_context_query(user_query, history)
        has_style_preference = self._extract_style_preference(style_context_query) is not None
        structured_query = style_context_query if self._has_structured_filter(user_query) or has_style_preference else search_query
        vector_docs = self.product_store.similarity_search(search_query, k=24)
        structured_docs, structured_filters = self._get_structured_product_docs(structured_query)
        docs = self._merge_unique_docs(structured_docs, vector_docs)
        print(
            f"🔍 Product Retrieval: {time.time() - t0:.2f}s "
            f"(Vector={len(vector_docs)}, Structured={len(structured_docs)}, Merged={len(docs)})"
        )
        self._chat_log(f"retrieval_count={len(docs)} | structured_filters={structured_filters}")
        
        if not docs:
            return self._build_chat_result("Xin lỗi, tôi không tìm thấy sản phẩm phù hợp với yêu cầu của bạn.", [], self._default_state_from_query(parsed_query, session_state), parsed_query)

        # 3. Re-ranking
        t1 = time.time()
        resolved_metas = [self._resolve_doc_metadata(d) for d in docs]
        candidate_indices = list(range(len(docs)))
        has_hard_structured_filters = self._has_hard_structured_filters(structured_filters)
        if has_hard_structured_filters:
            structured_matched_indices = [
                idx for idx, meta in enumerate(resolved_metas)
                if self._meta_passes_structured_filters(meta, structured_filters)
            ]
            if structured_matched_indices:
                candidate_indices = structured_matched_indices
                self._chat_log(
                    f"structured_filter applied filters={structured_filters} | kept={[self._compact_meta(resolved_metas[idx]) for idx in candidate_indices[:8]]}"
                )

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
        if parsed_query.intent in {"size_advice", "color_question"} and active_state.active_product_id:
            active_candidate_indices = [
                idx for idx in range(len(docs))
                if int(resolved_metas[idx].get("product_id", 0) or 0) == active_state.active_product_id
            ]
            for active_idx in reversed(active_candidate_indices):
                if active_idx not in candidate_indices:
                    candidate_indices.insert(0, active_idx)
                else:
                    candidate_indices.remove(active_idx)
                    candidate_indices.insert(0, active_idx)

        effective_rerank_top_k = max(RERANK_TOP_K, 8) if structured_filters else RERANK_TOP_K
        candidate_limit = max(1, min(effective_rerank_top_k, len(candidate_indices)))
        candidate_indices = candidate_indices[:candidate_limit]
        candidate_docs = [docs[idx] for idx in candidate_indices]
        candidate_metas = [resolved_metas[idx] for idx in candidate_indices]
        doc_texts = [
            self._resolve_doc_context(d, m)
            for d, m in zip(candidate_docs, candidate_metas)
        ]
        pairs = [[search_query, text] for text in doc_texts]

        if ENABLE_RERANK and pairs:
            scores = self.reranker.predict(pairs)
        else:
            scores = [0.0 for _ in pairs]

        scored_items = []
        for local_idx, score in enumerate(scores):
            idx = candidate_indices[local_idx]
            meta = resolved_metas[idx]
            token_score = self._token_overlap_score(search_query, meta)
            structured_score = self._structured_filter_score(meta, structured_filters)
            style_score = self._style_color_score(meta, structured_filters.get("style"))
            anchor_score = max(
                self._score_doc_anchor(user_query, meta),
                self._score_doc_anchor(search_query, meta),
                self._score_doc_anchor(anchor_text, meta),
            )
            boosted_score = float(score) + (0.60 * token_score) + (0.90 * anchor_score) + (1.20 * structured_score) + (0.70 * style_score)
            scored_items.append((idx, boosted_score, anchor_score, anchor_score >= 0.34))
        scored_items.sort(key=lambda item: item[1], reverse=True)
        self._chat_log(
            f"ranked_candidates={[{**self._compact_meta(resolved_metas[idx]), 'score': round(score, 4), 'anchor_score': round(anchor_score, 4), 'match': is_match} for idx, score, anchor_score, is_match in scored_items[:8]]}"
        )

        if not scored_items:
            return self._build_chat_result(
                "Xin lỗi, tôi không tìm thấy sản phẩm phù hợp với yêu cầu của bạn.",
                [],
                self._default_state_from_query(parsed_query, session_state),
                parsed_query,
            )

        primary_idx = self._select_primary_product_index(user_query, search_query, scored_items, resolved_metas)
        ordered_indices = []
        if primary_idx is not None:
            ordered_indices.append(primary_idx)
        ordered_indices.extend(idx for idx, *_ in scored_items if idx not in ordered_indices)

        explicit_anchor = bool(anchor_text.strip())
        if explicit_anchor:
            strict_indices = [
                idx for idx in ordered_indices
                if max(
                    self._score_doc_anchor(search_query, resolved_metas[idx]),
                    self._score_doc_anchor(anchor_text, resolved_metas[idx]),
                ) >= 0.34
            ]
            if strict_indices:
                ordered_indices = strict_indices
                self._chat_log(
                    f"strict_anchor_filter applied | kept={[self._compact_meta(resolved_metas[idx]) for idx in ordered_indices[:8]]}"
                )

        if parsed_query.intent in {"size_advice", "color_question"} and active_state.active_product_id:
            active_indices = [
                idx for idx in ordered_indices
                if int(resolved_metas[idx].get("product_id", 0) or 0) == active_state.active_product_id
            ]
            if active_indices:
                ordered_indices = active_indices
                self._chat_log(f"active_state_match product_id={active_state.active_product_id} -> kept_only_active_match")

        final_docs = [docs[idx] for idx in ordered_indices[:max(1, RERANK_TOP_K)]]
        answer_query = style_context_query if has_style_preference else user_query
        filtered_products = []
        seen_product_ids = set()
        for doc in final_docs:
            meta = self._resolve_doc_metadata(doc)
            product_id = int(meta.get("product_id", 0) or 0)
            if not product_id or product_id in seen_product_ids:
                continue
            seen_product_ids.add(product_id)
            filtered_products.append(
                ProductInfo(
                    id=product_id,
                    name=str(meta.get("name") or ""),
                    price=float(meta.get("price") or 0),
                    image_url=str(meta.get("primary_image_url") or meta.get("image_url") or ""),
                    category=str(meta.get("category_name") or meta.get("category") or ""),
                    brand=str(meta.get("brand_name") or meta.get("brand") or ""),
                    category_department=str(meta.get("category_department") or ""),
                    colors=self._extract_color_names(meta),
                    sizes=self._extract_size_names(meta),
                    averageRating=meta.get("averageRating") or meta.get("average_rating"),
                    reviewCount=meta.get("reviewCount") or meta.get("review_count"),
                    soldQuantity=meta.get("soldQuantity") or meta.get("sold_quantity"),
                )
            )

        print(f"Re-ranking: {time.time() - t1:.2f}s. Final Docs: {len(final_docs)}")
        self._chat_log(f"final_docs={[self._compact_meta(self._resolve_doc_metadata(doc)) for doc in final_docs]}")
        # Deterministic answer for size/color/style questions when metadata already has fields.

        combined_answer = self._build_combined_size_color_answer(answer_query, final_docs)
        if combined_answer:
            state = self._build_state_from_meta(parsed_query, self._resolve_doc_metadata(final_docs[0]), session_state)
            self._chat_log(f"branch=combined_size_color_answer | next_state={state.model_dump()}")
            return self._build_chat_result(combined_answer, filtered_products, state, parsed_query)

        preference_answer = self._build_preference_followup_answer(answer_query, final_docs)
        if preference_answer:
            state = self._build_state_from_meta(parsed_query, self._resolve_doc_metadata(final_docs[0]), session_state)
            self._chat_log(f"branch=preference_followup_answer | next_state={state.model_dump()}")
            return self._build_chat_result(preference_answer, filtered_products, state, parsed_query)

        size_answer = self._build_size_answer(answer_query, final_docs)
        if size_answer:
            state = self._build_state_from_meta(parsed_query, self._resolve_doc_metadata(final_docs[0]), session_state)
            self._chat_log(f"branch=size_answer | next_state={state.model_dump()}")
            return self._build_chat_result(size_answer, filtered_products, state, parsed_query)

        color_answer = self._build_color_answer(answer_query, final_docs)
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
    







