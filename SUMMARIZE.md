# Tổng Quan FShop AI Service

FShop AI là service FastAPI phụ trách các tính năng AI cho hệ thống fashion e-commerce:

- Chatbot RAG tư vấn sản phẩm và chính sách.
- Visual search bằng ảnh sản phẩm.
- Voice search và voice transcription.
- Recommendation dựa trên tương tác người dùng.
- Content moderation cho post, review, comment và livestream comment.
- Virtual try-on thông qua Hugging Face Space.

## Cấu Trúc Chính

- `app/main.py`: entrypoint FastAPI, khai báo lifespan và API endpoints.
- `app/core/config.py`: cấu hình env, database, Qdrant, model, voice và moderation.
- `app/models/schemas.py`: Pydantic schemas cho request/response.
- `app/services/rag_service.py`: chatbot RAG, intent parsing, retrieval, reranking và generation.
- `app/services/catalog_search_service.py`: tìm kiếm catalog từ text/voice query.
- `app/services/image_service.py`: embedding ảnh, visual search và recommendation.
- `app/services/voice_service.py`: nhận audio, transcription và voice search.
- `app/services/text_preprocessor.py`: chuẩn hóa và rule-based moderation.
- `app/services/phobert_service.py`: ML moderation bằng PhoBERT hoặc fallback heuristic.
- `app/services/moderation_engine.py`: kết hợp rule score và ML score để đưa ra quyết định.
- `app/services/vton_service.py`: client gọi virtual try-on.
- `data_scripts/`: script đồng bộ catalog/policy vào Qdrant.
- `training/`: notebook và dữ liệu phục vụ huấn luyện/đánh giá moderation model.

## Luồng Hoạt Động

1. Backend gọi AI service qua HTTP khi cần chatbot, search, moderation hoặc try-on.
2. AI service xử lý request bằng model/service tương ứng.
3. Với RAG/search, service truy vấn Qdrant và PostgreSQL-derived payload.
4. Với moderation, service trả về score, label, decision và priority để backend lưu log và cập nhật trạng thái nội dung.
5. Backend chịu trách nhiệm persist dữ liệu nghiệp vụ, còn AI service tập trung vào inference và retrieval.

## Điểm Cần Refactor Tiếp

- Tách `app/main.py` thành các router nhỏ theo domain.
- Tách `rag_service.py` thành các module nhỏ hơn: prompt, parser, retrieval, reranking, generation, product sync.
- Thay `print` bằng structured logging.
- Chuẩn hóa error handling và response mapping cho các endpoint.
- Thêm test tối thiểu cho moderation, voice utils và query intent parsing.
