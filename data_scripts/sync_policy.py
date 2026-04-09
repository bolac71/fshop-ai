import sys
import os
from importlib import import_module
from importlib.util import find_spec
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if find_spec("langchain_huggingface"):
    HuggingFaceEmbeddings = import_module(
        "langchain_huggingface"
    ).HuggingFaceEmbeddings
else:
    HuggingFaceEmbeddings = import_module(
        "langchain_community.embeddings"
    ).HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document

# Import config ta vừa sửa ở Bước 1
from app.core.config import (
    QDRANT_URL, 
    COLLECTION_POLICIES, 
    TEXT_EMBEDDING_MODEL
)

def sync_policies():
    print("--- 🚀 BẮT ĐẦU NẠP DỮ LIỆU CHÍNH SÁCH ---")

    policies_data = [
        # Chính sách đổi trả
        "Return Policy: Customers can return items within 30 days of purchase. Items must be unworn, unwashed, and with original tags attached. Refunds are processed within 5-7 business days to the original payment method.",
        
        # Chính sách vận chuyển
        "Shipping Policy: We offer Free Shipping for all orders base on your coupons. Standard shipping takes 3-5 business days. Express shipping (1-2 days) costs $20 extra. We ship via FedEx and DHL.",
        
        # Phương thức thanh toán
        "Payment Methods: We accept Credit Cards (Visa, MasterCard, Amex), PayPal, and Cash on Delivery (COD) for orders under $200.",
        
        # Giờ làm việc & Địa chỉ
        "Contact & Store Hours: Our physical store is located at 123 Fashion Street, Ho Chi Minh City. Open daily from 9:00 AM to 9:00 PM. Support Hotline: 1900-1234.",
        
        # Hướng dẫn chọn size
        "Size Guide: Our sizes follow standard International sizing (S, M, L, XL). If you are between sizes, we recommend sizing up for a comfortable fit."
    ]

    # 2. Kết nối Qdrant & Model Embedding
    print("Loading AI Model...")
    client = QdrantClient(url=QDRANT_URL)
    embeddings = HuggingFaceEmbeddings(model_name=TEXT_EMBEDDING_MODEL)

    # 3. Tạo lại Collection (Xóa cũ tạo mới cho sạch)
    if client.collection_exists(COLLECTION_POLICIES):
        client.delete_collection(COLLECTION_POLICIES)
    
    client.create_collection(
        collection_name=COLLECTION_POLICIES,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    # 4. Đóng gói dữ liệu thành Document
    documents = []
    for text in policies_data:
        doc = Document(page_content=text, metadata={"source": "policy_manual"})
        documents.append(doc)

    # 5. Đẩy vào Qdrant
    print(f"Đang nạp {len(documents)} chính sách vào Vector DB...")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_POLICIES,
        embedding=embeddings,
    )
    vector_store.add_documents(documents)
    
    print("--- NẠP CHÍNH SÁCH THÀNH CÔNG! ---")

if __name__ == "__main__":
    sync_policies()