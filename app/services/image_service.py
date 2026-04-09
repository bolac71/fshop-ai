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

        print("Image Service Ready!")

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
        Đây là chìa khóa của Cross-modal Search.
        """
        return self.model.encode(text).tolist()
    
    def search_by_text_vector(self, text_query: str, limit=5, filter=None):
        query_vector = self.encode_text(text_query)
        
        search_results = self.client.query_points(
            collection_name=COLLECTION_PRODUCT_IMAGE,
            query=query_vector,
            limit=limit,
            query_filter=filter,
            score_threshold=0.2 
        )
        return search_results.points

    def search_by_image(
        self,
        image_data: bytes,
        limit: int = 5,
        candidate_limit: int | None = None,
        score_threshold: float | None = None,
    ):
        query_vector = self._process_image(image_data)

        fetch_limit = candidate_limit or max(limit, 1)
        threshold = SEARCH_THRESHOLD if score_threshold is None else score_threshold

        search_results = self.client.query_points(
            collection_name=COLLECTION_PRODUCT_IMAGE,
            query=query_vector,
            limit=fetch_limit,
            score_threshold=threshold,
        )

        points = list(search_results.points)

        # Fallback: if threshold is too strict and returns too few points,
        # retry without threshold so API layer can still build top-k unique products.
        if len(points) < limit and threshold is not None:
            fallback_results = self.client.query_points(
                collection_name=COLLECTION_PRODUCT_IMAGE,
                query=query_vector,
                limit=fetch_limit,
            )

            seen_ids = {p.id for p in points}
            for point in fallback_results.points:
                if point.id not in seen_ids:
                    points.append(point)
                    seen_ids.add(point.id)

        return points

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
        self.client.delete(
            collection_name=COLLECTION_PRODUCT_IMAGE,
            points_selector=image_ids
        )
        return len(image_ids)
    
    def recommend_by_profile(self, interactions: list, limit=10):
        """
        Gợi ý sản phẩm dựa trên User Profile Vector (Weighted Centroid).
        Công thức: Vector_User = Sum(Vector_Item * Weight) / Sum(Weight)
        Với Weight = exp(-lambda * days_ago)
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
        LAMBDA_DECAY = 0.1 # Hệ số suy giảm

        # 3. Tính toán trọng số cho từng món hàng
        for item in interactions:
            if item.image_id in vector_map:
                vec = vector_map[item.image_id]
                
                weight = np.exp(-LAMBDA_DECAY * item.days_ago)
                
                vectors.append(vec)
                weights.append(weight)

        if not vectors:
            return []

        # 4. TÍNH VECTOR TRUNG BÌNH CÓ TRỌNG SỐ (USER PERSONA)
        np_vectors = np.array(vectors)
        np_weights = np.array(weights).reshape(-1, 1)

        weighted_sum = np.sum(np_vectors * np_weights, axis=0)
        total_weight = np.sum(np_weights)

        user_profile_vector = weighted_sum / total_weight

        # 5. Tìm kiếm sản phẩm gần nhất với User Profile Vector
        search_results = self.client.query_points(
            collection_name=COLLECTION_PRODUCT_IMAGE,
            query=user_profile_vector.tolist(),
            limit=limit,
            score_threshold=0.6 
        )

        if hasattr(search_results, 'points'):
             return search_results.points
        return []
