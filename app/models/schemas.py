from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Optional, Literal

# --- MODERATION V2 MODELS ---
ModerationContentType = Literal["post", "review", "post_comment", "livestream_comment"]
ModerationDecisionValue = Literal["approved", "flagged"]
ModerationPriorityValue = Literal["NORMAL", "HIGH"]


class ModerationLabel(BaseModel):
    label: str   # "toxic" | "spam" | "hate_speech" | "nsfw" | "off_topic"
    score: float


class ModerationV2Request(BaseModel):
    text: str
    content_type: ModerationContentType = "post_comment"
    content_id: Optional[int] = None
    user_id: Optional[int] = None


class ModerationV2Response(BaseModel):
    content_id: Optional[int]
    content_type: ModerationContentType
    rule_score: float
    ml_scores: List[ModerationLabel]
    final_score: float
    decision: ModerationDecisionValue
    priority: ModerationPriorityValue
    confidence: float
    processing_ms: int
    signals: Dict[str, Any]
    matched_patterns: List[str]


# --- CHAT MODELS ---
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str


class ChatSessionState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    active_product_id: Optional[int] = None
    active_product_name: Optional[str] = None
    active_category: Optional[str] = None
    active_brand: Optional[str] = None
    last_product_ids: List[int] = Field(default_factory=list)
    last_capability: Optional[str] = None
    last_task: Optional[str] = None
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    negative_preferences: List[str] = Field(default_factory=list)
    last_search_constraints: Dict[str, Any] = Field(default_factory=dict)
    last_intent: Optional[str] = None
    last_entities: Dict[str, Any] = Field(default_factory=dict)


class ParsedChatQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: Literal[
        "product_discovery",
        "product_qa",
        "product_compare",
        "product_search",
        "size_advice",
        "color_question",
        "recommend_products",
        "product_advice",
        "compare_products",
        "policy_qa",
        "order_qa",
        "clarify",
        "policy",
        "order",
        "small_talk",
        "unknown",
    ]
    search_query: str
    requires_context: bool = False
    confidence: float = 0.0
    entities: Dict[str, Any] = Field(default_factory=dict)
    follow_up: bool = False

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    history: List[ChatMessage] = Field(default_factory=list)
    user_id: Optional[int] = None
    session_state: Optional[ChatSessionState] = None

class ProductInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    name: str
    price: float
    image_url: str = ""
    category: str = ""
    brand: Optional[str] = ""
    category_department: Optional[str] = ""
    colors: List[str] = Field(default_factory=list)
    sizes: List[str] = Field(default_factory=list)
    averageRating: Optional[float] = None
    reviewCount: Optional[int] = None
    soldQuantity: Optional[int] = None

class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    products: List[ProductInfo]
    session_state: Optional[ChatSessionState] = None
    parsed_query: Optional[ParsedChatQuery] = None

# --- IMAGE SEARCH MODELS ---
class ImageSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int
    score: float
    image_url: str


class VoiceAsrMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language: str = "unknown"
    language_probability: float = 0.0
    avg_logprob: float = 0.0
    no_speech_probability: float = 0.0
    confidence: float = 0.0
    model: str = ""
    device: str = ""
    compute_type: str = ""


class VoiceSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transcribed_text: str
    products: List[ImageSearchResult]
    normalized_query: str = ""
    rewritten_query: str = ""
    intent: str = "product_search"
    filters: Dict[str, Any] = Field(default_factory=dict)
    asr_confidence: float = 0.0
    asr: VoiceAsrMetadata = Field(default_factory=VoiceAsrMetadata)
    latency_ms: int = 0
    search_debug: Optional[Dict[str, Any]] = None


class VoiceTranscriptionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transcribed_text: str
    normalized_query: str = ""
    rewritten_query: str = ""
    intent: str = "product_search"
    filters: Dict[str, Any] = Field(default_factory=dict)
    asr_confidence: float = 0.0
    asr: VoiceAsrMetadata = Field(default_factory=VoiceAsrMetadata)
    latency_ms: int = 0
    search_debug: Optional[Dict[str, Any]] = None

# --- VECTOR OPS MODELS ---
class IndexImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_id: int
    product_id: int   
    image_url: str
    source_type: Optional[str] = "product"
    variant_id: Optional[int] = None

class DeleteImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_ids: List[int]

class InteractionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_id: int  
    days_ago: float
    score: float = 1.0
    category_id: Optional[int] = None
    brand_id: Optional[int] = None
    color_id: Optional[int] = None
    size_id: Optional[int] = None

class RecommendRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")
    interactions: List[InteractionItem] 
    limit: int = 10

class ProductSyncRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    name: str
    description: Optional[str] = ""
    price: float
    image_url: Optional[str] = ""
    primary_image_url: Optional[str] = ""
    category: Optional[str] = "General"
    category_name: Optional[str] = ""
    category_department: Optional[str] = ""
    brand: Optional[str] = "Generic"
    brand_name: Optional[str] = ""
    attributes: Optional[str] = ""
    variants: Optional[List[dict]] = None

class ProductDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int

