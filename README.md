# FShop AI Service

Dịch vụ AI cho FShop, gồm các tính năng chính:

- Chatbot RAG (sản phẩm và chính sách)
- Tìm kiếm sản phẩm bằng ảnh và giọng nói
- Kiểm duyệt nội dung
- Virtual try-on

## 1. Yêu cầu

- Python 3.10 hoặc 3.11
- Docker Desktop (để chạy Qdrant)
- PostgreSQL đã có dữ liệu từ backend FShop

## 2. Cài đặt môi trường

Chạy tại thư mục gốc dự án:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Cấu hình biến môi trường

Tạo hoặc cập nhật file [.env](.env) với tối thiểu các biến sau:

```env
QDRANT_URL=http://localhost:6333
DB_NAME=fshop_db
DB_USER=username
DB_PASSWORD=123456
DB_HOST=localhost
DB_PORT=5432
GROQ_API_KEY=your_key
HF_TOKEN=your_key
HF_HUB_DISABLE_SYMLINKS_WARNING=1
TEXT_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
LLM_MODEL_NAME=llama-3.3-70b-versatile
LLM_REWRITE_MODEL=llama-3.1-8b-instant
SENTIMENT_MODEL_NAME=cardiffnlp/twitter-xlm-roberta-base-sentiment
VOICE_MODEL_SIZE=small
VOICE_LANGUAGE=vi
```

Bạn có thể chỉnh model hoặc collection trong [app/core/config.py](app/core/config.py) nếu cần.

## 4. Chạy Qdrant

```powershell
docker compose -f docker-compose.yaml up -d qdrant
```

Kiểm tra nhanh: [http://localhost:6333/collections](http://localhost:6333/collections)

## 5. Đồng bộ dữ liệu vào Qdrant

Catalog (Product + Variant + Image):

```powershell
python data_scripts/sync_catalog_qdrant.py --mode all --recreate
```

Chỉ text:

```powershell
python data_scripts/sync_catalog_qdrant.py --mode text --recreate
```

Chỉ image:

```powershell
python data_scripts/sync_catalog_qdrant.py --mode image --recreate
```

Policy (script riêng):

```powershell
python data_scripts/sync_policy.py
```

## 6. Chạy API (dev tự reload)

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 7. Ghi chú

- Script sync catalog chính: [data_scripts/sync_catalog_qdrant.py](data_scripts/sync_catalog_qdrant.py)
- Script sync policy: [data_scripts/sync_policy.py](data_scripts/sync_policy.py)
- Nếu lỗi kết nối Qdrant, kiểm tra container đã chạy và cổng `6333` đang mở.
- Nếu đổi `TEXT_EMBEDDING_MODEL`, cần chạy lại sync text (`--mode text --recreate`) để vector trong Qdrant khớp model mới.
- Trên Windows, cảnh báo symlink của Hugging Face là phổ biến và không làm hỏng service; có thể tắt bằng `HF_HUB_DISABLE_SYMLINKS_WARNING=1` hoặc bật Developer Mode để cache hiệu quả hơn.
