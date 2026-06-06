import io
import requests
from PIL import Image
from rembg import remove
from sentence_transformers import SentenceTransformer, models as st_models
from qdrant_client import QdrantClient, models as qd_models
import numpy as np
from app.core.config import (
    IMAGE_EMBEDDING_MODEL,
    MODEL_DISPLAY_NAME,
    COLLECTION_PRODUCT_IMAGE,
    IMAGE_VECTOR_SIZE,
    SEARCH_THRESHOLD
)

class ImageService:
    PRODUCT_IMAGE_ID_OFFSET = 1_000_000_000
    VARIANT_IMAGE_ID_OFFSET = 2_000_000_000

    def __init__(self, client: QdrantClient):
        self.client = client
        print(f"Loading Image Model: {MODEL_DISPLAY_NAME}")
        print(f"Model Path/Name: {IMAGE_EMBEDDING_MODEL}")

        try:
            # Standardized loading (works for both models)
            self.model = SentenceTransformer(IMAGE_EMBEDDING_MODEL)
            print(f"✅ {MODEL_DISPLAY_NAME} loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading {MODEL_DISPLAY_NAME}: {str(e)}")
            raise

        self._ensure_image_collection()

        print("Image Service Ready!")

    def _ensure_image_collection(self) -> None:
        """Ensure image collection exists so search/upsert won't fail on first run."""
        exists = self.client.collection_exists(COLLECTION_PRODUCT_IMAGE)
        if exists:
            return

        self.client.create_collection(
            collection_name=COLLECTION_PRODUCT_IMAGE,
            vectors_config=qd_models.VectorParams(
                size=IMAGE_VECTOR_SIZE,
                distance=qd_models.Distance.COSINE,
            ),
        )
        print(
            f"Created missing Qdrant collection '{COLLECTION_PRODUCT_IMAGE}' "
            f"(size={IMAGE_VECTOR_SIZE})."
        )

    def _process_image(self, image_data: bytes) -> list:
        """Helper: Đọc ảnh -> Xóa phông -> Embed"""
        original_image = Image.open(io.BytesIO(image_data))
        
        # Xóa phông để vector chính xác hơn vào quần áo
        no_bg_image = remove(original_image)
        clean_image = Image.new("RGB", no_bg_image.size, (255, 255, 255))
        clean_image.paste(no_bg_image, mask=no_bg_image.split()[3])
        
        return self.model.encode(clean_image).tolist()

    def encode_text(self, text: str) -> list:
        """
        Chuyển mô tả văn bản thành vector cùng không gian với ảnh.
        Vietnamese SBERT: Optimized for Vietnamese semantic understanding.
        Normalize embeddings for cosine similarity in Qdrant.
        """
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def search_by_text_vector(self, text_query: str, limit=5, filter=None):
        query_vector = self.encode_text(text_query)
        
        search_results = self.client.query_points(
            collection_name=COLLECTION_PRODUCT_IMAGE,
            query=query_vector,
            limit=limit,
            query_filter=filter,
            score_threshold=0.62
        )
        return search_results.points

    def search_by_image(
        self,
        image_data: bytes,
        limit: int = 5,
        candidate_limit: int | None = None,
        score_threshold: float | None = None,
    ):
        self._ensure_image_collection()
        query_vector = self._process_image(image_data)

        fetch_limit = candidate_limit or max(limit, 1)
        threshold = SEARCH_THRESHOLD if score_threshold is None else score_threshold

        search_results = self.client.query_points(
            collection_name=COLLECTION_PRODUCT_IMAGE,
            query=query_vector,
            limit=fetch_limit,
            score_threshold=threshold,
        )

        return list(search_results.points)

    def _build_point_id(self, image_id: int, source_type: str = "product") -> int:
        if source_type == "variant":
            return self.VARIANT_IMAGE_ID_OFFSET + image_id
        return self.PRODUCT_IMAGE_ID_OFFSET + image_id

    def upsert_image_vector(
        self,
        image_id: int,
        product_id: int,
        image_url: str,
        source_type: str = "product",
        variant_id: int | None = None,
    ):
        self._ensure_image_collection()
        # Tải ảnh từ URL
        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            raise Exception("Cannot download image from URL")
        
        vector = self._process_image(response.content)
        point_id = self._build_point_id(image_id, source_type)
        
        point = qd_models.PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "product_id": product_id,
                "image_url": image_url,
                "source_type": source_type,
                "source_image_id": image_id,
                "variant_id": variant_id,
            }
        )
        self.client.upsert(
            collection_name=COLLECTION_PRODUCT_IMAGE,
            points=[point]
        )
        return True

    def delete_vectors(self, image_ids: list):
        self._ensure_image_collection()
        self.client.delete(
            collection_name=COLLECTION_PRODUCT_IMAGE,
            points_selector=image_ids
        )
        return len(image_ids)
    
    def recommend_by_profile(self, interactions: list, limit=10):
        """
        Gợi ý sản phẩm Hybrid (Vector + Metadata).
        1. Tính User Profile Vector dựa trên hành vi (Weighted Centroid).
        2. Tìm kiếm ứng viên từ Qdrant (Lấy nhiều hơn limit để re-rank).
        3. Tái sắp xếp (Re-rank) dựa trên Metadata (Brand, Category, Color, Size).
        """
        if not interactions:
            return []

        # 1. Lấy danh sách ID để query Qdrant
        target_ids = [item.image_id for item in interactions]
        
        # 2. Lấy vectors từ Qdrant
        records = self.client.retrieve(
            collection_name=COLLECTION_PRODUCT_IMAGE,
            ids=target_ids,
            with_vectors=True 
        )
        
        vector_map = {point.id: point.vector for point in records}

        vectors = []
        weights = []
        LAMBDA_DECAY = 0.1 # Hệ số suy giảm theo thời gian

        # 3. Tính toán trọng số cho từng món hàng (Kết hợp hành vi + thời gian)
        for item in interactions:
            if item.image_id in vector_map:
                vec = vector_map[item.image_id]
                
                # Trọng số = exp(-lambda * days_ago) * interaction_score
                # interaction_score được gửi từ NestJS (1, 3, 5, 10)
                time_weight = np.exp(-LAMBDA_DECAY * item.days_ago)
                total_weight = time_weight * getattr(item, 'score', 1.0)
                
                vectors.append(vec)
                weights.append(total_weight)

        if not vectors:
            return []

        # 4. TÍNH VECTOR TRUNG BÌNH (USER PERSONA)
        np_vectors = np.array(vectors)
        np_weights = np.array(weights).reshape(-1, 1)

        weighted_sum = np.sum(np_vectors * np_weights, axis=0)
        total_weight = np.sum(np_weights)
        user_profile_vector = weighted_sum / total_weight

        # 5. TÌM KIẾM ỨNG VIÊN (Lấy nhiều hơn để re-rank)
        candidate_limit = limit * 4 
        search_results = self.client.query_points(
            collection_name=COLLECTION_PRODUCT_IMAGE,
            query=user_profile_vector.tolist(),
            limit=candidate_limit,
            score_threshold=0.55 
        )
        
        points = search_results.points if hasattr(search_results, 'points') else []
        print(f"Found {len(points)} candidates for re-ranking based on profile vector.")
        if not points:
            return []

        # 6. RE-RANKING DỰA TRÊN METADATA
        # Xây dựng profile metadata của User (lấy các giá trị phổ biến nhất trong interactions)
        user_brands = [it.brand_id for it in interactions if getattr(it, 'brand_id', None)]
        user_categories = [it.category_id for it in interactions if getattr(it, 'category_id', None)]
        user_colors = [it.color_id for it in interactions if getattr(it, 'color_id', None)]
        user_sizes = [it.size_id for it in interactions if getattr(it, 'size_id', None)]

        def calculate_boost(point_payload):
            boost = 0.0
            # Trùng Brand/Category (Trọng số cao: 0.2)
            if point_payload.get("brand_id") in user_brands:
                boost += 0.2
            if point_payload.get("category_id") in user_categories:
                boost += 0.2
            # Trùng Color/Size (Trọng số thấp hơn: 0.1)
            if point_payload.get("color_id") in user_colors:
                boost += 0.1
            if point_payload.get("size_id") in user_sizes:
                boost += 0.1
            return boost

        # Áp dụng Re-rank
        scored_points = []
        for p in points:
            original_score = p.score
            boost = calculate_boost(p.payload)
            p.score = original_score + boost
            scored_points.append(p)

        # Sắp xếp lại theo score mới
        scored_points.sort(key=lambda x: x.score, reverse=True)
        print(f"Found {len(scored_points)} candidates for re-ranking based on profile vector.")
        for p in scored_points[:limit]:
            print(f"Candidate ID: {p.id}, Original Score: {p.score - boost:.4f}, Boost: {boost:.4f}, Final Score: {p.score:.4f}")
        return scored_points[:limit]

