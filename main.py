from fastapi import FastAPI, HTTPException, UploadFile, File 
from pydantic import BaseModel
from typing import List, Optional
import json
import requests
import io
from rembg import remove
from PIL import Image

from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
QDRANT_URL = "http://localhost:6333" 
COLLECTION_NAME = "fashion_products"
VECTOR_SIZE = 384
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:7b"

IMAGE_COLLECTION = "fashion_images"
IMAGE_MODEL_NAME = "clip-ViT-B-32"

SCORE_THRESHOLD = 0.65

# 1. Initialize FastAPI Application
app = FastAPI(title="Fashion AI Service")


# 2. Initialize Embedding Model
print("Loading Embedding Model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
print("Embedding Model is ready!")

print(f"Loading CLIP Model: {IMAGE_MODEL_NAME}...")
clip_model = SentenceTransformer(IMAGE_MODEL_NAME)

print("Tất cả Model đã sẵn sàng!")

# 3. Initialize Qdrant Client
client = QdrantClient(url=QDRANT_URL)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)


# 4. Prompt Template
RAG_TEMPLATE = """
You are a professional fashion sales assistant for "FShop".
You have access to the following product list (Context) in English.

---
CONTEXT (Product Data):
{context}
---

USER QUESTION: {question}

INSTRUCTIONS:
1. **Language Rule**: 
   - If the User Question is in **Vietnamese**, you MUST answer in **natural, native Vietnamese**.
   - If the User Question is in **English**, answer in **English**.
   - **NEVER** use Russian, Chinese, or other languages.

2. **Analyze the Context**: 
   - The Context contains search results from the database. Some might be IRRELEVANT (e.g., user asks for "Men" but context has "Kids", user asks for "Leather" but context has "Cotton").
   - **STRICTLY FILTER**: Only mention products that actually match the User Question's key constraints (Gender, Color, Material, Type). 
   - If a product in the context doesn't match, IGNORE it.
3. **Formulate Answer**:
   - If multiple valid products are found, list them briefly (e.g., "I found a few options: A ($price) and B ($price)...").
   - Don't just focus on one if there are others.
   - Keep the product names in English.
   - Answer in the **SAME LANGUAGE** as the User Question (Vietnamese or English).
4. **Content Rule**:
   - Base your recommendation ONLY on the provided CONTEXT.
   - Do not invent features not listed.
   - Keep the original English names of the products (e.g., "Lightweight Bomber"), do not translate product names.
   
4. **No Match Case**:
   - If NO products in the Context match the user's criteria (even if the Context is not empty), apologize and say you don't have that specific item.

5. **Tone**:
   - Be helpful, concise, and encouraging (like a real shop assistant).
   - Mention the price in the user's currency format if possible (or keep $).

ANSWER:
"""

prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# --- DATA MODELS ---
class ChatRequest(BaseModel):
    question: str

class ProductInfo(BaseModel):
    id: int
    name: str
    price: float
    image_url: str
    category: str

class ChatResponse(BaseModel):
    answer: str
    products: List[ProductInfo]

class ImageSearchResult(BaseModel):
    product_id: int
    score: float
    image_url: str

class IndexImageRequest(BaseModel):
    image_id: int     
    product_id: int   
    image_url: str     

class DeleteImageRequest(BaseModel):
    image_ids: List[int]

# --- Upsert Vector ---
@app.post("/vectors/upsert")
async def upsert_vector(req: IndexImageRequest):
    try:
        print(f"🔄 Indexing Image ID: {req.image_id}...")
        
        # 1. Tải ảnh từ URL (Cloudinary)
        response = requests.get(req.image_url, timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Cannot download image from URL")
            
        image = Image.open(io.BytesIO(response.content))

        # 2. Tạo Vector
        vector = clip_model.encode(image).tolist()

        # 3. Upsert vào Qdrant
        point = PointStruct(
            id=req.image_id, 
            vector=vector,
            payload={
                "product_id": req.product_id,
                "image_url": req.image_url
            }
        )
        
        client.upsert(
            collection_name=IMAGE_COLLECTION,
            points=[point]
        )
        return {"status": "success", "image_id": req.image_id}

    except Exception as e:
        print(f"Error indexing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Delete Vector ---
@app.post("/vectors/delete")
async def delete_vectors(req: DeleteImageRequest):
    try:
        print(f"🗑 Deleting Vectors: {req.image_ids}...")
        client.delete(
            collection_name=IMAGE_COLLECTION,
            points_selector=req.image_ids
        )
        return {"status": "deleted", "count": len(req.image_ids)}
    except Exception as e:
        print(f"Error deleting: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/search/image", response_model=List[ImageSearchResult])
async def search_by_image(file: UploadFile = File(...)):
    """
    Nhận file ảnh -> Tạo Vector -> Tìm sản phẩm giống nhất
    """
    print(f"Received request to search for image: {file.filename}")
    
    try:
        # 1. Đọc ảnh từ file upload
        image_data = await file.read()
        original_image = Image.open(io.BytesIO(image_data))

        print("Applying Background Removal (rembg)...")

        no_bg_image = remove(original_image)

        clean_image = Image.new("RGB", no_bg_image.size, (255, 255, 255))
        clean_image.paste(no_bg_image, mask=no_bg_image.split()[3])
        
        # 2. Tạo Vector (Embedding) dùng CLIP
        query_vector = clip_model.encode(clean_image).tolist()
        
        # 3. Tìm kiếm trong Qdrant (Collection Fashion Images)
        search_results = client.query_points(
            collection_name=IMAGE_COLLECTION,
            query=query_vector,
            limit=5,
            score_threshold=SCORE_THRESHOLD,
        )
        
        # 4. Map kết quả trả về
        results = []
        seen_ids = set() 
        
        if not search_results.points:
            print("Ảnh không giống sản phẩm nào (Score thấp).")
            return [] 
        

        for hit in search_results.points:
            pid = hit.payload.get("product_id")
            print(f"--> Found ID {pid} | Score: {hit.score}")
            if pid not in seen_ids:
                results.append(ImageSearchResult(
                    product_id=pid,
                    score=hit.score, 
                    image_url=hit.payload.get("image_url", "")
                ))
                seen_ids.add(pid)
                
        return results

    except Exception as e:
        print(f"Lỗi Image Search: {e}")
        raise HTTPException(status_code=500, detail="Không thể xử lý hình ảnh này")

@app.post("/chat/ask", response_model=ChatResponse)
async def ask_fashion_ai(request: ChatRequest):
    """
    RAG Flow:
    1. Nhận câu hỏi -> Embedding
    2. Search Qdrant -> Lấy Top 4 sản phẩm liên quan
    3. Ghép thông tin sản phẩm vào Prompt
    4. Gửi cho LLM (Qwen) -> Nhận câu trả lời
    5. Trả về kết quả kèm Metadata sản phẩm
    """

    user_query = request.question
    print(f"User asking: {user_query}")

    # Step 1: Search products in vector DB (Retrieval)
    # k=4 means top-4 similar products
    docs = vector_store.similarity_search(user_query, k=4)

    if not docs:
        return ChatResponse(
            answer="Sorry, I couldn't find any products for you.",
            products=[]
        )

    # Step 2: Prepare context for LLM
    context_text = ""
    recommended_products = []
    for doc in docs:
        meta = doc.metadata
        price_str = f"{meta.get('price', 0):,.0f}"

        product_desc = (
            f"- Name: {meta.get('name')}\n"
            f"  Brand: {meta.get('brand')}\n"
            f"  Price: {price_str}\n"
            f"  Description: {doc.page_content}\n" # page_content chứa full text cũ
        )
        context_text += product_desc + "\n"

        # prepare objects to return for frontend
        recommended_products.append(ProductInfo(
            id=meta.get("product_id", 0),
            name=meta.get("name", "Unknown"),
            price=float(meta.get("price", 0)),
            image_url=meta.get("image_url", ""),
            category=meta.get("category", "")
        ))  
    
    # Step 3: Create prompt for LLM
    chain = prompt | llm | StrOutputParser()

    try:
        ai_answer = chain.invoke({
            "context": context_text,
            "question": user_query
        })
        ai_answer = ai_answer.replace("Answer:", "").strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        raise HTTPException(status_code=500, detail="AI processing failed")

    # Step 4: Return response
    return ChatResponse(
        answer=ai_answer,
        products=recommended_products
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)