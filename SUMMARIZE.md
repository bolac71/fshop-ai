# 📘 Tài liệu Phân tích Hệ thống FShop AI

Dự án **fshop-ai** là một dịch vụ backend chuyên biệt về AI, cung cấp các tính năng thông minh cho hệ thống thương mại điện tử FShop. Dịch vụ này được xây dựng bằng **FastAPI** và tích hợp nhiều mô hình học máy (Machine Learning) tiên tiến.

## 1. 📂 Cấu trúc thư mục chính

*   `app/`: Thư mục chứa mã nguồn chính của ứng dụng.
    *   `core/`: Cấu hình hệ thống (`config.py`).
    *   `models/`: Định nghĩa các schema dữ liệu (Pydantic models).
    *   `services/`: **Trái tim của hệ thống**, chứa logic xử lý các tác vụ AI:
        *   `rag_service.py`: Xử lý Chatbot hỏi đáp về sản phẩm (RAG).
        *   `image_service.py`: Xử lý tìm kiếm bằng hình ảnh (CLIP model).
        *   `voice_service.py`: Xử lý tìm kiếm bằng giọng nói.
        *   `moderation_engine.py`: Kiểm duyệt nội dung tự động (PhoBERT).
        *   `catalog_search_service.py`: Logic tìm kiếm tổng hợp.
    *   `main.py`: Điểm khởi chạy API chính.
*   `data_scripts/`: Các script dùng để đồng bộ dữ liệu từ Database vào Vector Database (Qdrant).
*   `training/`: (Nếu có) Chứa các script hoặc dữ liệu liên quan đến việc huấn luyện/tinh chỉnh mô hình.
*   `main.py` (root): Thường là bản thử nghiệm hoặc phiên bản standalone cũ.

## 2. 🚀 Các tính năng cốt lõi

### 💬 1. Chatbot AI (RAG Flow)
*   **Cơ chế:** Sử dụng kỹ thuật **Retrieval-Augmented Generation (RAG)**.
*   **Quy trình:** Khi người dùng hỏi -> Hệ thống tìm kiếm các sản phẩm liên quan trong Qdrant -> Gửi thông tin sản phẩm đó kèm câu hỏi vào LLM (như Llama-3 hoặc Qwen) -> Trả về câu trả lời tự nhiên.
*   **Hỗ trợ:** Đa ngôn ngữ (Tiếng Việt & Tiếng Anh).

### 🔍 2. Tìm kiếm bằng Hình ảnh (Visual Search)
*   **Cơ chế:** Sử dụng mô hình **CLIP** (Contrastive Language-Image Pre-training).
*   **Quy trình:** Ảnh tải lên được xóa phông (`rembg`) -> Chuyển thành Vector -> So sánh độ tương đồng trong Qdrant để tìm ra các sản phẩm có kiểu dáng, màu sắc tương tự.

### 🎙️ 3. Tìm kiếm bằng Giọng nói (Voice Search)
*   **Cơ chế:** Chuyển đổi giọng nói thành văn bản (ASR) và sau đó thực hiện tìm kiếm.
*   **Công nghệ:** Thường sử dụng mô hình Whisper hoặc các thư viện Speech-to-Text tương đương.

### 🛡️ 4. Kiểm duyệt nội dung (Content Moderation)
*   **Cơ chế:** Kết hợp giữa luật (Rule-based) và AI (Machine Learning).
*   **Công nghệ:** Sử dụng mô hình **PhoBERT** (mô hình ngôn ngữ tối ưu cho Tiếng Việt) để phân loại bình luận/nội dung là "nhạy cảm", "xúc phạm" hay "hợp lệ".

### 📊 5. Gợi ý sản phẩm (Recommendation)
*   Dựa trên lịch sử tương tác của người dùng để gợi ý các sản phẩm phù hợp thông qua Vector Similarity.

## 3. 🛠 Công nghệ sử dụng

| Công nghệ | Mục đích |
| :--- | :--- |
| **FastAPI** | Framework xây dựng API tốc độ cao. |
| **Qdrant** | Vector Database dùng để lưu trữ và tìm kiếm vector (ảnh, text). |
| **LangChain** | Framework quản lý luồng xử lý cho LLM và RAG. |
| **Hugging Face** | Cung cấp các mô hình như CLIP, PhoBERT, Sentence Transformers. |
| **Ollama / Groq** | Chạy các mô hình ngôn ngữ lớn (LLM). |
| **Rembg** | Công cụ AI dùng để xóa phông nền ảnh tự động. |

## ⚙️ Luồng hoạt động (Workflow)

1.  **Đồng bộ dữ liệu:** Chạy các script trong `data_scripts/` để lấy dữ liệu sản phẩm từ PostgreSQL, chuyển thành Vector và đẩy vào Qdrant.
2.  **Xử lý Request:** Người dùng gửi yêu cầu (Text/Image/Voice) qua API.
3.  **AI Processing:** Tùy vào loại request, Service tương ứng sẽ được gọi để xử lý (nhúng vector, gọi LLM, v.v.).
4.  **Kết quả:** Trả về dữ liệu JSON bao gồm câu trả lời của AI và danh sách sản phẩm gợi ý.