import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from qdrant_client import QdrantClient

from app.core.config import QDRANT_URL
from app.services.rag_service import RagService


CASES = [
    ("Shop có bán áo thun không?", {"Áo thun nam", "Áo thun nữ"}),
    ("shop có bán áo sơ mi không?", {"Áo sơ mi nam", "Áo sơ mi nữ"}),
    ("Tôi muốn mua quần jeans nữ", {"Quần jeans nữ"}),
    ("Có giày thể thao nam không?", {"Giày thể thao nam"}),
]

STYLE_CASES = [
    (
        "Mình muốn màu nào ngầu ngầu, cool, nhìn chất chơi",
        {"den", "xanh navy", "than chi", "xam ghi", "xam", "xanh duong"},
    ),
]


def main() -> int:
    client = QdrantClient(url=QDRANT_URL)
    service = RagService.__new__(RagService)
    service.client = client
    service.chat_debug_enabled = False
    service._catalog_index = None
    service._catalog_index_loaded_at = 0.0
    service._style_profiles = None

    failures = []
    for query, expected_categories in CASES:
        docs, filters = service._get_structured_product_docs(query, limit=5)
        categories = {
            (doc.metadata or {}).get("category_name")
            for doc in docs[:3]
            if (doc.metadata or {}).get("category_name")
        }
        ok = bool(categories) and categories.issubset(expected_categories)
        print(f"\nQuery: {query}")
        print(f"Filters: {filters}")
        print(f"Top categories: {sorted(categories)}")
        print("OK" if ok else "FAILED")
        if not ok:
            failures.append((query, categories, expected_categories))

    for query, expected_colors in STYLE_CASES:
        docs, filters = service._get_structured_product_docs(query, limit=5)
        top_colors = []
        for doc in docs[:5]:
            top_colors.extend(service._extract_color_names(doc.metadata or {}))
        normalized_colors = {service._normalize_text(color) for color in top_colors}
        ok = bool(filters.get("style")) and bool(normalized_colors & expected_colors)
        print(f"\nStyle query: {query}")
        print(f"Filters: {filters}")
        print(f"Top colors: {sorted(normalized_colors)}")
        print("OK" if ok else "FAILED")
        if not ok:
            failures.append((query, normalized_colors, expected_colors))

    client.close()
    if failures:
        print("\nRetrieval evaluation failed:")
        for query, categories, expected in failures:
            print(f"- {query}: got={sorted(categories)} expected_subset={sorted(expected)}")
        return 1

    print("\nAll retrieval evaluation cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
