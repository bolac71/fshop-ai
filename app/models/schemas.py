from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Optional, Literal

class ModerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    context: Optional[str] = "product_review"

class ModerationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    is_safe: bool          
    label: str            
    confidence: float    
    status: str           
    processing_mode: str   

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
    last_intent: Optional[str] = None
    last_entities: Dict[str, Any] = Field(default_factory=dict)


class ParsedChatQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: Literal["product_search", "size_advice", "color_question", "policy", "order", "unknown"]
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
class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    interactions: List[InteractionItem] 

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

# --- VIRTUAL TRY-ON MODELS (IDM-VTON) ---
class VirtualTryonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    """Request model for virtual try-on endpoint (IDM-VTON)"""
    category: str = "upper_body"  # upper_body, lower_body, dress
    garment_des: str = ""  # Garment description (optional)
    seed: int = 42  # Random seed for reproducibility
    steps: int = 30  # Inference steps (20-50)
    crop: bool = False  # Whether to crop images
    force_dc: bool = False  # Force DC optimization
    mask_only: bool = False  # Return only mask
    resize: bool = True  # Whether to resize image for optimal processing
    max_dimension: int = 768  # Max width/height for resizing

class VirtualTryonResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    """Response model for virtual try-on endpoint"""
    success: bool
    message: str
    image_base64: Optional[str] = None  # Base64 encoded result image
    category: str
    seed: int
    processing_time: Optional[float] = None  # Processing time in seconds

class VirtualTryonBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    """Request model for batch virtual try-on"""
    garments: List[dict]  # List of garments with image bytes and categories

class VirtualTryonBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    """Response model for batch virtual try-on"""
    success: bool
    message: str
    results: List[VirtualTryonResponse]
    total_time: Optional[float] = None