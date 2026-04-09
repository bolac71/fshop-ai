import argparse
from importlib import import_module
from importlib.util import find_spec
import json
import os
import sys
import time
from decimal import Decimal
from io import BytesIO
from typing import Any, Dict, List, Sequence, Tuple


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import psycopg2
import requests
from PIL import Image, UnidentifiedImageError

if find_spec("langchain_huggingface"):
    HuggingFaceEmbeddings = import_module(
        "langchain_huggingface"
    ).HuggingFaceEmbeddings
else:
    HuggingFaceEmbeddings = import_module(
        "langchain_community.embeddings"
    ).HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from app.core.config import (
    COLLECTION_PRODUCT_IMAGE,
    COLLECTION_PRODUCT_TEXT,
    IMAGE_EMBEDDING_MODEL,
    IMAGE_VECTOR_SIZE,
    POSTGRES_CONFIG,
    QDRANT_URL,
    TEXT_EMBEDDING_MODEL,
    TEXT_VECTOR_SIZE,
)


TEXT_SQL = """
WITH sold AS (
    SELECT variant_id, COALESCE(SUM(quantity), 0) AS sold_quantity
    FROM inventory_transactions
    WHERE type = 'EXPORT'
    GROUP BY variant_id
)
SELECT
    p.id,
    p.name,
    p.description,
    p.price,
    p.average_rating,
    p.review_count,
    b.id AS brand_id,
    b.name AS brand_name,
    b.slug AS brand_slug,
    c.id AS category_id,
    c.name AS category_name,
    c.slug AS category_slug,
    c.department AS category_department,
    COALESCE((
        SELECT pi.image_url
        FROM product_images pi
        WHERE pi.product_id = p.id
          AND pi.is_active = TRUE
        ORDER BY pi.id ASC
        LIMIT 1
    ), '') AS primary_image_url,
    COALESCE((
        SELECT ARRAY_AGG(DISTINCT co.name ORDER BY co.name)
        FROM product_variants pv
        JOIN colors co ON co.id = pv.color_id
        WHERE pv.product_id = p.id
          AND pv.is_active = TRUE
    ), ARRAY[]::text[]) AS color_names,
    COALESCE((
        SELECT ARRAY_AGG(DISTINCT sz.name ORDER BY sz.name)
        FROM product_variants pv
        JOIN sizes sz ON sz.id = pv.size_id
        WHERE pv.product_id = p.id
          AND pv.is_active = TRUE
    ), ARRAY[]::text[]) AS size_names,
    COALESCE((
        SELECT JSON_AGG(
            JSON_BUILD_OBJECT(
                'variant_id', pv.id,
                'sku', pv.sku,
                'color_id', pv.color_id,
                'color_name', co.name,
                'color_hex', co.hex_code,
                'size_id', pv.size_id,
                'size_name', sz.name,
                'size_type_id', sz.size_type_id,
                'stock_quantity', COALESCE(inv.quantity, 0),
                'sold_quantity', COALESCE(sold.sold_quantity, 0),
                'image_url', pv.image_url
            )
            ORDER BY pv.id
        )
        FROM product_variants pv
        LEFT JOIN colors co ON co.id = pv.color_id
        LEFT JOIN sizes sz ON sz.id = pv.size_id
        LEFT JOIN inventories inv ON inv.variant_id = pv.id
        LEFT JOIN sold ON sold.variant_id = pv.id
        WHERE pv.product_id = p.id
          AND pv.is_active = TRUE
    ), '[]'::json) AS variants_json
FROM products p
LEFT JOIN brands b ON b.id = p.brand_id
LEFT JOIN categories c ON c.id = p.category_id
WHERE p.is_active = TRUE
ORDER BY p.id ASC;
"""


PRODUCT_IMAGE_SQL = """
SELECT
    pi.id AS source_image_id,
    p.id AS product_id,
    pi.image_url,
    pi.public_id,
    p.name AS product_name,
    p.price AS product_price,
    b.name AS brand_name,
    c.name AS category_name,
    NULL::int AS variant_id,
    ''::text AS color_name,
    ''::text AS size_name,
    'product'::text AS source_type
FROM product_images pi
JOIN products p ON p.id = pi.product_id
LEFT JOIN brands b ON b.id = p.brand_id
LEFT JOIN categories c ON c.id = p.category_id
WHERE p.is_active = TRUE
  AND pi.is_active = TRUE
  AND pi.image_url IS NOT NULL

UNION ALL

SELECT
    pv.id AS source_image_id,
    p.id AS product_id,
    pv.image_url,
    pv.public_id,
    p.name AS product_name,
    p.price AS product_price,
    b.name AS brand_name,
    c.name AS category_name,
    pv.id AS variant_id,
    COALESCE(co.name, '') AS color_name,
    COALESCE(sz.name, '') AS size_name,
    'variant'::text AS source_type
FROM product_variants pv
JOIN products p ON p.id = pv.product_id
LEFT JOIN brands b ON b.id = p.brand_id
LEFT JOIN categories c ON c.id = p.category_id
LEFT JOIN colors co ON co.id = pv.color_id
LEFT JOIN sizes sz ON sz.id = pv.size_id
WHERE p.is_active = TRUE
  AND pv.is_active = TRUE
  AND pv.image_url IS NOT NULL;
"""


def _to_plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def _load_json_if_needed(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _connect_db():
    return psycopg2.connect(**POSTGRES_CONFIG)


def _fetch_rows(sql: str) -> List[Tuple[Any, ...]]:
    conn = _connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()
    finally:
        conn.close()


def _ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    recreate: bool,
) -> None:
    exists = client.collection_exists(collection_name)
    if exists and recreate:
        client.delete_collection(collection_name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )


def _build_product_text(row: Sequence[Any]) -> Tuple[str, Dict[str, Any]]:
    product_id = int(row[0])
    name = row[1] or ""
    description = row[2] or ""
    price = float(row[3]) if row[3] is not None else 0.0
    average_rating = float(row[4]) if row[4] is not None else 0.0
    review_count = int(row[5]) if row[5] is not None else 0

    brand_id = row[6]
    brand_name = row[7] or ""
    brand_slug = row[8] or ""

    category_id = row[9]
    category_name = row[10] or ""
    category_slug = row[11] or ""
    category_department = row[12] or ""

    primary_image_url = row[13] or ""
    color_names = _to_plain(row[14] or [])
    size_names = _to_plain(row[15] or [])
    variants = _to_plain(_load_json_if_needed(row[16])) or []

    variant_lines: List[str] = []
    variant_summary: List[Dict[str, Any]] = []
    for variant in variants:
        color = variant.get("color_name") or "Unknown color"
        size = variant.get("size_name") or "Unknown size"
        sku = variant.get("sku") or ""
        stock = int(variant.get("stock_quantity") or 0)
        sold = int(variant.get("sold_quantity") or 0)
        sku_part = f" | SKU: {sku}" if sku else ""
        variant_lines.append(f"{color} / {size}{sku_part} | stock={stock} | sold={sold}")
        variant_summary.append(
            {
                "variant_id": variant.get("variant_id"),
                "sku": sku,
                "color_id": variant.get("color_id"),
                "color_name": color,
                "color_hex": variant.get("color_hex") or "",
                "size_id": variant.get("size_id"),
                "size_name": size,
                "size_type_id": variant.get("size_type_id"),
                "stock_quantity": stock,
                "sold_quantity": sold,
                "image_url": variant.get("image_url") or "",
            }
        )

    variants_text = "; ".join(variant_lines) if variant_lines else "No active variants"
    colors_text = ", ".join(color_names) if color_names else "N/A"
    sizes_text = ", ".join(size_names) if size_names else "N/A"

    anchor_terms = sorted(
        {
            term.strip().lower()
            for term in [
                name,
                brand_name,
                category_name,
                category_department,
                *color_names,
                *size_names,
            ]
            if term and str(term).strip()
        }
    )

    search_text = " ".join(
        part
        for part in [
            name,
            brand_name,
            category_name,
            category_department,
            description,
            colors_text,
            sizes_text,
            variants_text,
        ]
        if part
    )

    page_content = (
        f"Product: {name}. "
        f"Brand: {brand_name}. "
        f"Category: {category_name}. "
        f"Department: {category_department}. "
        f"Description: {description}. "
        f"Price: {price:.2f}. "
        f"Rating: {average_rating:.1f}/5 from {review_count} reviews. "
        f"Available colors: {colors_text}. "
        f"Available sizes: {sizes_text}. "
        f"Variants: {variants_text}."
    )

    payload: Dict[str, Any] = {
        "product_id": product_id,
        "name": name,
        "description": description,
        "price": price,
        "average_rating": average_rating,
        "review_count": review_count,
        "brand_id": int(brand_id) if brand_id is not None else None,
        "brand_name": brand_name,
        "brand_slug": brand_slug,
        "category_id": int(category_id) if category_id is not None else None,
        "category_name": category_name,
        "category_slug": category_slug,
        "category_department": category_department,
        "primary_image_url": primary_image_url,
        "color_names": color_names,
        "size_names": size_names,
        "variants": variants,
        "variant_summary": variant_summary,
        "anchor_terms": anchor_terms,
        "search_text": search_text,
        "schema_version": 2,
    }
    return page_content, payload


def sync_product_text(client: QdrantClient, recreate: bool, batch_size: int = 64) -> int:
    print("[TEXT] Fetching products from PostgreSQL...")
    rows = _fetch_rows(TEXT_SQL)
    if not rows:
        print("[TEXT] No active products found.")
        return 0

    print(f"[TEXT] Found {len(rows)} active products.")
    _ensure_collection(client, COLLECTION_PRODUCT_TEXT, TEXT_VECTOR_SIZE, recreate)

    print(f"[TEXT] Loading embedding model: {TEXT_EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=TEXT_EMBEDDING_MODEL)

    text_buffer: List[str] = []
    payload_buffer: List[Dict[str, Any]] = []
    point_buffer: List[models.PointStruct] = []
    total = 0

    def flush_points() -> int:
        if not text_buffer:
            return 0
        vectors = embeddings.embed_documents(text_buffer)
        for payload, vector in zip(payload_buffer, vectors):
            point_buffer.append(
                models.PointStruct(
                    id=int(payload["product_id"]),
                    vector=vector,
                    payload=payload,
                )
            )
        client.upsert(collection_name=COLLECTION_PRODUCT_TEXT, points=point_buffer)
        synced = len(point_buffer)
        text_buffer.clear()
        payload_buffer.clear()
        point_buffer.clear()
        return synced

    for row in rows:
        page_content, payload = _build_product_text(row)
        text_buffer.append(page_content)
        payload_buffer.append(payload)

        if len(text_buffer) >= batch_size:
            total += flush_points()
            print(f"[TEXT] Upserted {total} vectors...")

    if text_buffer:
        total += flush_points()

    print(f"[TEXT] Done. Total vectors: {total}")
    return total


def _download_image_with_retry(
    session: requests.Session,
    url: str,
    timeout: int,
    retries: int,
) -> Image.Image:
    last_error: Exception | None = None
    for _ in range(max(retries, 1)):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except (requests.RequestException, UnidentifiedImageError) as exc:
            last_error = exc
            time.sleep(0.5)
    if last_error is None:
        raise RuntimeError("Unknown image download error")
    raise last_error


def sync_product_images(
    client: QdrantClient,
    recreate: bool,
    timeout: int = 12,
    retries: int = 3,
    batch_size: int = 32,
) -> int:
    print("[IMAGE] Fetching product and variant images from PostgreSQL...")
    rows = _fetch_rows(PRODUCT_IMAGE_SQL)
    if not rows:
        print("[IMAGE] No active images found.")
        return 0

    print(f"[IMAGE] Found {len(rows)} image records.")
    _ensure_collection(client, COLLECTION_PRODUCT_IMAGE, IMAGE_VECTOR_SIZE, recreate)

    print(f"[IMAGE] Loading image embedding model: {IMAGE_EMBEDDING_MODEL}")
    image_model = SentenceTransformer(IMAGE_EMBEDDING_MODEL)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }
    )

    uploaded = 0
    skipped = 0
    point_buffer: List[models.PointStruct] = []
    seen_point_ids: set[int] = set()

    # Qdrant in this setup accepts only unsigned integer or UUID point IDs.
    # Keep IDs deterministic and collision-free across product/variant sources.
    PRODUCT_IMAGE_ID_OFFSET = 1_000_000_000
    VARIANT_IMAGE_ID_OFFSET = 2_000_000_000

    for row in rows:
        source_image_id = int(row[0])
        product_id = int(row[1])
        image_url = row[2]
        public_id = row[3] or ""
        product_name = row[4] or ""
        product_price = float(row[5]) if row[5] is not None else 0.0
        brand_name = row[6] or ""
        category_name = row[7] or ""
        variant_id = int(row[8]) if row[8] is not None else None
        color_name = row[9] or ""
        size_name = row[10] or ""
        source_type = row[11] or "product"

        if not image_url:
            skipped += 1
            continue

        if source_type == "variant":
            point_id = VARIANT_IMAGE_ID_OFFSET + source_image_id
        else:
            point_id = PRODUCT_IMAGE_ID_OFFSET + source_image_id

        if point_id in seen_point_ids:
            continue

        try:
            image = _download_image_with_retry(session, image_url, timeout=timeout, retries=retries)
            vector = image_model.encode(image).tolist()

            payload = {
                "product_id": product_id,
                "source_image_id": source_image_id,
                "source_type": source_type,
                "variant_id": variant_id,
                "image_url": image_url,
                "public_id": public_id,
                "product_name": product_name,
                "product_price": product_price,
                "brand_name": brand_name,
                "category_name": category_name,
                "color_name": color_name,
                "size_name": size_name,
            }

            point_buffer.append(models.PointStruct(id=point_id, vector=vector, payload=payload))
            seen_point_ids.add(point_id)

            if len(point_buffer) >= batch_size:
                client.upsert(collection_name=COLLECTION_PRODUCT_IMAGE, points=point_buffer)
                uploaded += len(point_buffer)
                print(f"[IMAGE] Upserted {uploaded} vectors...")
                point_buffer = []

        except Exception as exc:
            skipped += 1
            print(f"[IMAGE] Skip image {point_id}: {exc}")

    if point_buffer:
        client.upsert(collection_name=COLLECTION_PRODUCT_IMAGE, points=point_buffer)
        uploaded += len(point_buffer)

    print(f"[IMAGE] Done. Uploaded: {uploaded}, Skipped: {skipped}")
    return uploaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync FShop product catalog (text + image) from PostgreSQL to Qdrant."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "text", "image"],
        default="all",
        help="Which data to sync.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate target collections before syncing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of vectors per Qdrant upsert batch.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=12,
        help="Timeout (seconds) when downloading images.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry attempts for each image download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=== FShop Catalog Sync to Qdrant ===")
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Mode: {args.mode} | Recreate: {args.recreate} | Batch size: {args.batch_size}")

    client = QdrantClient(url=QDRANT_URL)

    text_count = 0
    image_count = 0

    if args.mode in ("all", "text"):
        text_count = sync_product_text(
            client=client,
            recreate=args.recreate,
            batch_size=max(args.batch_size, 1),
        )

    if args.mode in ("all", "image"):
        image_count = sync_product_images(
            client=client,
            recreate=args.recreate,
            timeout=max(args.request_timeout, 1),
            retries=max(args.retries, 1),
            batch_size=max(args.batch_size, 1),
        )

    print("=== Sync Summary ===")
    print(f"Text vectors synced: {text_count}")
    print(f"Image vectors synced: {image_count}")


if __name__ == "__main__":
    main()
