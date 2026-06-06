# FShop AI Service

Dịch vụ AI cho hệ thống FShop, gồm các nhóm chức năng chính:

- Chatbot RAG cho tư vấn sản phẩm và chính sách cửa hàng.
- Tìm kiếm sản phẩm bằng hình ảnh và giọng nói.
- Gợi ý sản phẩm dựa trên hồ sơ tương tác.
- Kiểm duyệt nội dung cho bài viết, đánh giá, bình luận và livestream.
- Virtual try-on thông qua Hugging Face Space.

## Yêu Cầu

- Python 3.10 hoặc 3.11.
- Docker Desktop để chạy Qdrant.
- PostgreSQL đã có dữ liệu từ backend FShop.

## Cài Đặt

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Cấu Hình

Tạo hoặc cập nhật file `.env` tại thư mục gốc dự án:

```env
QDRANT_URL=http://localhost:6333
DB_NAME=fshop_db
DB_USER=username
DB_PASSWORD=123456
DB_HOST=localhost
DB_PORT=5432
GROQ_API_KEY=your_key
HF_TOKEN=your_key
TEXT_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
TEXT_VECTOR_SIZE=384
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
AI_ENABLE_RERANK=false
AI_RERANK_TOP_K=4
LLM_MODEL_NAME=llama-3.3-70b-versatile
LLM_REWRITE_MODEL=llama-3.1-8b-instant
VOICE_MODEL_SIZE=medium
VOICE_LANGUAGE=vi
VOICE_DEVICE=cpu
VOICE_COMPUTE_TYPE=int8
VOICE_BEAM_SIZE=5
VOICE_VAD_FILTER=true
```

Các biến cấu hình chính được khai báo trong `app/core/config.py`.

## Chạy Qdrant

```powershell
docker compose -f docker-compose.yaml up -d qdrant
```

Kiểm tra nhanh:

```text
http://localhost:6333/collections
```

## Đồng Bộ Dữ Liệu Vào Qdrant

Đồng bộ toàn bộ catalog:

```powershell
python data_scripts/sync_catalog_qdrant.py --mode all --recreate
```

Chỉ đồng bộ text:

```powershell
python data_scripts/sync_catalog_qdrant.py --mode text --recreate
```

Chỉ đồng bộ image:

```powershell
python data_scripts/sync_catalog_qdrant.py --mode image --recreate
```

Đồng bộ policy:

```powershell
python data_scripts/sync_policy.py
```

Kiểm tra nhanh retrieval chatbot sau khi đồng bộ catalog/policy:

```powershell
python data_scripts/evaluate_chat_retrieval.py
```

Script này kiểm tra các truy vấn danh mục phổ biến như áo thun, áo sơ mi, quần jeans nữ, giày thể thao nam và một case style preference như "cool/ngầu" để đảm bảo top results không bị lệch category hoặc bỏ qua tín hiệu phong cách.

## Chạy API

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Ghi Chú

- Nếu đổi `TEXT_EMBEDDING_MODEL` hoặc `TEXT_VECTOR_SIZE`, cần sync lại text vector.
- Nếu đổi image model hoặc vector size, cần sync lại image vector.
- Trên Windows, cảnh báo symlink của Hugging Face là bình thường. Có thể đặt `HF_HUB_DISABLE_SYMLINKS_WARNING=1` hoặc bật Developer Mode.
