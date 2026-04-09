from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from qdrant_client import QdrantClient
import uvicorn
import requests
import time
import io

from app.core.config import QDRANT_URL
from app.models.schemas import (
    ChatRequest, ChatResponse, ImageSearchResult, 
    IndexImageRequest, DeleteImageRequest, RecommendRequest, 
    ProductSyncRequest, ProductDeleteRequest,
    ModerationRequest, ModerationResponse,
    VirtualTryonRequest, VirtualTryonResponse, VirtualTryonBatchRequest,
    VirtualTryonBatchResponse
)
from app.services.image_service import ImageService
from app.services.rag_service import RagService
from app.services.voice_service import VoiceService
from app.services.sentiment_service import SentimentService
from app.services.virtual_tryon_service import VirtualTryonService 

# --- GLOBAL VARIABLES ---
qdrant_client = None
image_service = None
rag_service = None
voice_service = None
sentiment_service = None
virtual_tryon_service = None

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Fashion AI Service (Lifespan)...")
    
    global qdrant_client, image_service, rag_service, voice_service, sentiment_service, virtual_tryon_service
    
    try:
        # Init Qdrant
        qdrant_client = QdrantClient(url=QDRANT_URL)
        
        # Init Services
        image_service = ImageService(qdrant_client)
        rag_service = RagService(qdrant_client)
        voice_service = VoiceService()
        
        # Init Sentiment Service
        sentiment_service = SentimentService()
        
        # Init Virtual Try-on Service (FREE via Gradio/HF Spaces)
        try:
            virtual_tryon_service = VirtualTryonService()
        except Exception as e:
            print(f"⚠️  Virtual Try-on Service initialization failed: {e}")
            print("   Virtual try-on endpoints will not be available")
            virtual_tryon_service = None
        
        print("✅ All Services Initialized Successfully!")
        
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        # Không raise error để server vẫn chạy (debug), nhưng thực tế nên handle kỹ
    
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

@app.post("/search/voice")
async def search_by_voice(file: UploadFile = File(...)):
    if not voice_service or not image_service:
        raise HTTPException(status_code=503, detail="Services not ready")
        
    data = await voice_service.process_voice_search(file, image_service)
    
    results = []
    seen_ids = set()
    for hit in data["results"]:
        pid = hit.payload.get("product_id")
        if pid not in seen_ids:
            results.append(ImageSearchResult(
                product_id=pid,
                score=hit.score,
                image_url=hit.payload.get("image_url", "")
            ))
            seen_ids.add(pid)
            
    return {
        "transcribed_text": data["text"], 
        "products": results
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

@app.post("/moderate/content", response_model=ModerationResponse)
async def moderate_content(req: ModerationRequest):
    """
    API Kiểm duyệt nội dung Hybrid (Local + LLM)
    """
    if sentiment_service is None:
        print("❌ CRITICAL: SentimentService is None inside endpoint!")
        raise HTTPException(status_code=503, detail="Sentiment Service is not initialized yet.")

    text = req.text.strip()
    if not text:
        return ModerationResponse(
            text=text, is_safe=True, label="NEUTRAL", 
            confidence=1.0, status="approved", processing_mode="none"
        )

    # BƯỚC 1: Fast Check (Local multilingual model + keyword)
    analysis = sentiment_service.analyze(text)
    
    final_label = analysis["label"]
    confidence = analysis["score"]
    is_toxic = analysis["is_toxic"]
    mode = "fast_local"

    print(f"🧐 Local Analysis: '{text}' -> {final_label} ({confidence:.2f}) | Toxic: {is_toxic}")

    # BƯỚC 2: Deep Check (LLM) nếu cần
    if analysis["requires_llm_check"]:
        print(f"🤔 Ambiguous content, invoking LLM: '{text}'")
        if rag_service:
            llm_result = rag_service.moderate_content(text)
            
            final_label = llm_result["sentiment"]
            is_toxic = llm_result["is_toxic"]
            if is_toxic:
                final_label = "NEGATIVE"
            confidence = 0.95 
            mode = "deep_llm"
            print(f"🤖 LLM Verdict: {final_label} (Toxic: {is_toxic})")
        else:
            print("⚠️ RAG Service missing, skipping LLM check")

    # BƯỚC 3: Quyết định Status
    status = "approved"
    if is_toxic:
        status = "rejected"
    elif final_label == "NEGATIVE":
        status = "approved" 
    
    return ModerationResponse(
        text=text,
        is_safe=not is_toxic,
        label=final_label,
        confidence=confidence,
        status=status,
        processing_mode=mode
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

# --- VIRTUAL TRY-ON ENDPOINTS ---
@app.post("/tryon/apply-image", response_class=StreamingResponse)
async def apply_virtual_tryon_image(
    person_file: UploadFile = File(...),
    garment_file: UploadFile = File(...),
    category: str = "upper_body",
    garment_des: str = "",
    seed: int = 42,
    steps: int = 30,
    crop: bool = False,
    force_dc: bool = False,
    mask_only: bool = False,
    resize: bool = True,
    max_dimension: int = 768
):
    """
    Apply virtual try-on effect and return the result as an IMAGE FILE (can preview/download in Swagger).
    
    - **person_file**: Image file of the person (human_img)
    - **garment_file**: Image file of the garment/clothing (garm_img)
    - **category**: Garment category - "upper_body", "lower_body", "dresses"
    - **garment_des**: Description of the garment (optional)
    - **seed**: Random seed for reproducibility (default: 42)
    - **steps**: Number of inference steps (default: 30, range: 20-50)
    - **crop**: Whether to crop images (default: false)
    - **force_dc**: Force DC optimization (default: false)
    - **mask_only**: Return only mask (default: false)
    - **resize**: Auto-resize images (default: true)
    - **max_dimension**: Max image dimension (default: 768)
    
    Returns PNG image file that you can view and download directly in Swagger.
    """
    if not virtual_tryon_service:
        raise HTTPException(
            status_code=503, 
            detail="Virtual Try-on Service not available. Service initialization failed."
        )
    
    try:
        start_time = time.time()
        
        # Read files
        person_image = await person_file.read()
        garment_image = await garment_file.read()
        
        if not person_image or not garment_image:
            raise HTTPException(status_code=400, detail="Invalid image files")
        
        print(f"📸 Virtual Try-on Request:")
        print(f"   Person: {person_file.filename} ({len(person_image)} bytes)")
        print(f"   Garment: {garment_file.filename} ({len(garment_image)} bytes)")
        
        # Create request data from parameters
        request_data = VirtualTryonRequest(
            category=category,
            garment_des=garment_des,
            seed=seed,
            steps=steps,
            crop=crop,
            force_dc=force_dc,
            mask_only=mask_only,
            resize=resize,
            max_dimension=max_dimension
        )
        
        # Resize images if requested
        if request_data.resize:
            person_image = virtual_tryon_service.resize_image_for_tryon(
                person_image, 
                request_data.max_dimension
            )
            garment_image = virtual_tryon_service.resize_image_for_tryon(
                garment_image, 
                request_data.max_dimension
            )
            print(f"   ✓ Images resized to max {request_data.max_dimension}px")
        
        # Apply try-on
        result = await virtual_tryon_service.tryon_garment(
            person_image=person_image,
            garment_image=garment_image,
            category=request_data.category,
            garment_des=request_data.garment_des,
            seed=request_data.seed,
            steps=request_data.steps,
            crop=request_data.crop,
            force_dc=request_data.force_dc,
            mask_only=request_data.mask_only
        )
        
        processing_time = time.time() - start_time
        print(f"⏱️  Processing time: {processing_time:.2f}s")
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        result["image"].save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Return as image file with proper content-type
        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={
                "Content-Disposition": f"inline; filename=tryon_result_{int(time.time())}.png"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Virtual try-on error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Virtual try-on failed: {str(e)}")


@app.post("/tryon/apply", response_model=VirtualTryonResponse)
async def apply_virtual_tryon(
    person_file: UploadFile = File(...),
    garment_file: UploadFile = File(...),
    category: str = "upper_body",
    garment_des: str = "",
    seed: int = 42,
    steps: int = 30,
    crop: bool = False,
    force_dc: bool = False,
    mask_only: bool = False,
    resize: bool = True,
    max_dimension: int = 768
):
    """
    Apply virtual try-on effect and return JSON with base64 encoded image (for API integration).
    
    - **person_file**: Image file of the person (human_img)
    - **garment_file**: Image file of the garment/clothing (garm_img)
    - **category**: Garment category - "upper_body", "lower_body", "dresses"
    - **garment_des**: Description of the garment (optional)
    - **seed**: Random seed for reproducibility (default: 42)
    - **steps**: Number of inference steps (default: 30, range: 20-50)
    - **crop**: Whether to crop images (default: false)
    - **force_dc**: Force DC optimization (default: false)
    - **mask_only**: Return only mask (default: false)
    - **resize**: Auto-resize images (default: true)
    - **max_dimension**: Max image dimension (default: 768)
    
    Returns JSON with base64 encoded result image (use /tryon/apply-image for direct image preview).
    """
    if not virtual_tryon_service:
        raise HTTPException(
            status_code=503, 
            detail="Virtual Try-on Service not available. Service initialization failed."
        )
    
    try:
        start_time = time.time()
        
        # Read files
        person_image = await person_file.read()
        garment_image = await garment_file.read()
        
        if not person_image or not garment_image:
            raise HTTPException(status_code=400, detail="Invalid image files")
        
        print(f"📸 Virtual Try-on Request:")
        print(f"   Person: {person_file.filename} ({len(person_image)} bytes)")
        print(f"   Garment: {garment_file.filename} ({len(garment_image)} bytes)")
        
        # Create request data from parameters
        request_data = VirtualTryonRequest(
            category=category,
            garment_des=garment_des,
            seed=seed,
            steps=steps,
            crop=crop,
            force_dc=force_dc,
            mask_only=mask_only,
            resize=resize,
            max_dimension=max_dimension
        )
        
        # Resize images if requested
        if request_data.resize:
            person_image = virtual_tryon_service.resize_image_for_tryon(
                person_image, 
                request_data.max_dimension
            )
            garment_image = virtual_tryon_service.resize_image_for_tryon(
                garment_image, 
                request_data.max_dimension
            )
            print(f"   ✓ Images resized to max {request_data.max_dimension}px")
        
        # Apply try-on
        result = await virtual_tryon_service.tryon_garment(
            person_image=person_image,
            garment_image=garment_image,
            category=request_data.category,
            garment_des=request_data.garment_des,
            seed=request_data.seed,
            steps=request_data.steps,
            crop=request_data.crop,
            force_dc=request_data.force_dc,
            mask_only=request_data.mask_only
        )
        
        processing_time = time.time() - start_time
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        
        return VirtualTryonResponse(
            success=True,
            message=result["message"],
            image_base64=result["base64"],
            category=request_data.category,
            seed=request_data.seed,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Virtual try-on error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Virtual try-on failed: {str(e)}")

@app.post("/tryon/batch", response_model=VirtualTryonBatchResponse)
async def apply_virtual_tryon_batch(
    person_file: UploadFile = File(...),
    garments: list = None
):
    """
    Apply virtual try-on with multiple garments for batch processing.
    
    garments format (JSON):
    [
        {
            "image_url": "https://...",
            "garment_type": "top",
            "name": "optional_name",
            "strength": 1.0
        },
        ...
    ]
    """
    if not virtual_tryon_service:
        raise HTTPException(
            status_code=503, 
            detail="Virtual Try-on Service not available"
        )
    
    try:
        start_time = time.time()
        
        person_image = await person_file.read()
        if not person_image:
            raise HTTPException(status_code=400, detail="Invalid person image")
        
        if not garments or not isinstance(garments, list):
            raise HTTPException(status_code=400, detail="Invalid garments list")
        
        print(f"👚 Batch Virtual Try-on Request: {len(garments)} garments")
        
        # Resize person image once
        person_image = virtual_tryon_service.resize_image_for_tryon(person_image)
        
        # Process each garment
        results = []
        for idx, garment in enumerate(garments):
            try:
                garment_url = garment.get("image_url")
                if not garment_url:
                    results.append(VirtualTryonResponse(
                        success=False,
                        message="Missing image_url",
                        garment_type=garment.get("garment_type", "unknown"),
                        strength=garment.get("strength", 1.0)
                    ))
                    continue
                
                # Download garment image
                response = requests.get(garment_url, timeout=10)
                if response.status_code != 200:
                    results.append(VirtualTryonResponse(
                        success=False,
                        message=f"Failed to download garment image: {response.status_code}",
                        garment_type=garment.get("garment_type", "unknown"),
                        strength=garment.get("strength", 1.0)
                    ))
                    continue
                
                garment_image = response.content
                
                # Apply try-on
                result = await virtual_tryon_service.tryon_garment(
                    person_image=person_image,
                    garment_image=garment_image,
                    garment_type=garment.get("garment_type", "top"),
                    strength=garment.get("strength", 1.0)
                )
                
                results.append(VirtualTryonResponse(
                    success=result["success"],
                    message=result["message"],
                    image_base64=result["base64"] if result["success"] else None,
                    garment_type=garment.get("garment_type", "top"),
                    strength=garment.get("strength", 1.0)
                ))
                
            except Exception as e:
                print(f"Error processing garment {idx}: {str(e)}")
                results.append(VirtualTryonResponse(
                    success=False,
                    message=f"Error: {str(e)}",
                    garment_type=garment.get("garment_type", "unknown"),
                    strength=garment.get("strength", 1.0)
                ))
        
        processing_time = time.time() - start_time
        
        return VirtualTryonBatchResponse(
            success=all(r.success for r in results),
            message=f"Processed {len(results)} garments",
            results=results,
            total_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Batch virtual try-on error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.get("/tryon/garment-types")
async def get_supported_garment_types():
    """Get list of supported garment types for virtual try-on"""
    if not virtual_tryon_service:
        return {"garment_types": []}
    
    try:
        types = await virtual_tryon_service.get_supported_garment_types()
        return {"garment_types": types}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=False)