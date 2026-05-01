from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
import uvicorn

from app.core.config import QDRANT_URL
import time

from app.models.schemas import (
    ChatRequest, ChatResponse, ImageSearchResult, VoiceSearchResponse,
    VoiceTranscriptionResponse,
    IndexImageRequest, DeleteImageRequest, RecommendRequest,
    ProductSyncRequest, ProductDeleteRequest,
    ModerationV2Request, ModerationV2Response,
)
from app.services.image_service import ImageService
from app.services.rag_service import RagService
from app.services.voice_service import VoiceService
from app.services.catalog_search_service import CatalogSearchService
from app.services.text_preprocessor import TextPreprocessor
from app.services.phobert_service import PhoBERTModerationService
from app.services.moderation_engine import ModerationEngine

# --- GLOBAL VARIABLES ---
qdrant_client = None
image_service = None
rag_service = None
voice_service = None
catalog_search_service = None
text_preprocessor = TextPreprocessor()
phobert_service = PhoBERTModerationService()
moderation_engine = ModerationEngine()

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Fashion AI Service (Lifespan)...")
    
    global qdrant_client, image_service, rag_service, voice_service, catalog_search_service

    try:
        # Init Qdrant
        qdrant_client = QdrantClient(url=QDRANT_URL)

        # Init Services
        image_service = ImageService(qdrant_client)
        rag_service = RagService(qdrant_client)
        catalog_search_service = CatalogSearchService(
            qdrant_client,
            rag_service=rag_service,
            image_service=image_service,
        )
        voice_service = VoiceService()

        print("✅ All Services Initialized Successfully!")
        
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        import traceback
        traceback.print_exc()
    
    yield # Điểm phân cách: Server bắt đầu nhận request tại đây
    
    # 2. SHUTDOWN
    print("🛑 Shutting down Fashion AI Service...")
    if qdrant_client:
        qdrant_client.close()
    print("Bye bye!")

# --- INIT APP ---
app = FastAPI(title="Fashion AI Service Pro", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporary: allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENDPOINTS ---

@app.post("/search/voice", response_model=VoiceSearchResponse)
async def search_by_voice(file: UploadFile = File(...)):
    if not voice_service or not image_service:
        raise HTTPException(status_code=503, detail="Services not ready")
        
    data = await voice_service.process_voice_search(file, image_service, catalog_search_service)
    
    if data.get("products"):
        results = [ImageSearchResult(**product) for product in data["products"]]
    else:
        results = []
        seen_ids = set()
        for hit in data.get("results", []):
            pid = hit.payload.get("product_id")
            if pid not in seen_ids:
                results.append(ImageSearchResult(
                    product_id=pid,
                    score=hit.score,
                    image_url=hit.payload.get("image_url", "")
                ))
                seen_ids.add(pid)
            
    return {
        "transcribed_text": data.get("transcribed_text", data.get("text", "")),
        "products": results,
        "normalized_query": data.get("normalized_query", ""),
        "rewritten_query": data.get("rewritten_query", ""),
        "intent": data.get("intent", "product_search"),
        "filters": data.get("filters", {}),
        "asr_confidence": data.get("asr_confidence", 0.0),
        "asr": data.get("asr", {}),
        "latency_ms": data.get("latency_ms", 0),
        "search_debug": data.get("search_debug"),
    }

@app.post("/transcribe/voice", response_model=VoiceTranscriptionResponse)
async def transcribe_voice(file: UploadFile = File(...)):
    if not voice_service:
        raise HTTPException(status_code=503, detail="Voice service not ready")

    data = await voice_service.transcribe_upload(file)
    return {
        "transcribed_text": data.get("transcribed_text", ""),
        "normalized_query": data.get("normalized_query", ""),
        "rewritten_query": data.get("rewritten_query", ""),
        "intent": data.get("intent", "product_search"),
        "filters": data.get("filters", {}),
        "asr_confidence": data.get("asr_confidence", 0.0),
        "asr": data.get("asr", {}),
        "latency_ms": data.get("latency_ms", 0),
        "search_debug": data.get("search_debug"),
    }

@app.post("/search/image", response_model=list[ImageSearchResult])
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=30),
):
    print(f"📸 Received image search: {file.filename}")
    if not image_service:
        raise HTTPException(status_code=503, detail="Image Service not initialized")
        
    try:
        content = await file.read()
        candidate_limit = max(top_k * 8, top_k)
        points = image_service.search_by_image(
            content,
            limit=top_k,
            candidate_limit=candidate_limit,
        )
        
        results = []
        seen_ids = set()
        for hit in points:
            pid = hit.payload.get("product_id")
            if pid not in seen_ids:
                results.append(ImageSearchResult(
                    product_id=pid,
                    score=hit.score,
                    image_url=hit.payload.get("image_url", "")
                ))
                seen_ids.add(pid)
            if len(results) >= top_k:
                break
        return results
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Image processing failed")

@app.post("/chat/ask", response_model=ChatResponse)
async def ask_fashion_ai(request: ChatRequest):
    print(f"User Query: {request.question} - UserID: {request.user_id}")
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG Service not initialized")
        
    result = await rag_service.chat(
        request.question,
        request.history,
        user_id=request.user_id,
        session_state=request.session_state,
    )
    return ChatResponse(**result)


@app.post("/moderate/v2", response_model=ModerationV2Response)
async def moderate_v2(req: ModerationV2Request):
    """
    Full moderation pipeline: text preprocessing → rule scoring → PhoBERT ML → decision engine.
    Returns soft-flag decision; the calling backend is responsible for persisting the result.
    """
    start_ms = int(time.time() * 1000)

    text = req.text.strip()
    if not text:
        return ModerationV2Response(
            content_id=req.content_id,
            content_type=req.content_type,
            rule_score=0.0,
            ml_scores=[],
            final_score=0.0,
            decision="approved",
            priority="NORMAL",
            confidence=1.0,
            processing_ms=0,
            signals={},
            matched_patterns=[],
        )

    processed = text_preprocessor.preprocess(text)
    ml_labels = phobert_service.predict(processed.normalized)
    decision = moderation_engine.decide(processed.rule_score, ml_labels)

    return ModerationV2Response(
        content_id=req.content_id,
        content_type=req.content_type,
        rule_score=processed.rule_score,
        ml_scores=ml_labels,
        final_score=decision.final_score,
        decision=decision.decision,
        priority=decision.priority,
        confidence=decision.confidence,
        processing_ms=int(time.time() * 1000) - start_ms,
        signals=processed.signals,
        matched_patterns=processed.matched_patterns,
    )


# --- ADMIN ENDPOINTS ---
@app.post("/vectors/upsert")
async def upsert_vector(req: IndexImageRequest):
    try:
        image_service.upsert_image_vector(
            req.image_id,
            req.product_id,
            req.image_url,
            source_type=req.source_type or "product",
            variant_id=req.variant_id,
        )
        return {"status": "success", "image_id": req.image_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vectors/delete")
async def delete_vectors(req: DeleteImageRequest):
    try:
        count = image_service.delete_vectors(req.image_ids)
        return {"status": "deleted", "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/profile-based")
async def recommend_profile_based(request: RecommendRequest):
    if not image_service: return {"product_ids": []}
    try:
        hits = image_service.recommend_by_profile(request.interactions)
        product_ids = []
        seen = set()
        for hit in hits:
            payload = hit.payload 
            if not isinstance(payload, dict) and hasattr(payload, 'dict'):
                 payload = payload.dict()
            if payload:
                pid = payload.get("product_id")
                if pid and pid not in seen:
                    product_ids.append(pid)
                    seen.add(pid)
        return {"product_ids": product_ids}
    except Exception as e:
        print(f"Recommendation Error: {e}")
        return {"product_ids": []}

@app.post("/products/sync")
async def sync_product_endpoint(req: ProductSyncRequest):
    try:
        rag_service.upsert_product(req)
        return {"status": "success", "message": f"Product {req.id} synced."}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to sync product")

@app.post("/products/remove")
async def remove_product_endpoint(req: ProductDeleteRequest):
    try:
        rag_service.delete_product(req.product_id)
        return {"status": "success", "message": f"Product {req.product_id} removed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to remove product")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=False)
