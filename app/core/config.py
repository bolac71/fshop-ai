import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


# --- DATABASE CONFIG ---
POSTGRES_CONFIG = {
    "dbname": _env("DB_NAME", "fshop_db"),
    "user": _env("DB_USER", "username"),
    "password": _env("DB_PASSWORD", "123456"),
    "host": _env("DB_HOST", "localhost"),
    "port": _env("DB_PORT", "5432"),
}


# --- QDRANT CONFIG ---
QDRANT_URL = _env("QDRANT_URL", "http://localhost:6333")


# --- MODEL NAMES ---
# Multilingual embedding model performs better for Vietnamese queries.
TEXT_EMBEDDING_MODEL = _env(
    "TEXT_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
TEXT_VECTOR_SIZE = _env_int("TEXT_VECTOR_SIZE", 384)
RERANKER_MODEL = _env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

MODEL_TYPE_BASELINE = "baseline"
MODEL_TYPE_FINETUNED = "finetuned"

USE_MODEL_TYPE = _env("USE_MODEL_TYPE", MODEL_TYPE_BASELINE).strip().lower()

BASELINE_MODEL_NAME = _env("BASELINE_IMAGE_MODEL", "clip-ViT-B-32")
FINETUNED_MODEL_PATH = _env(
    "FINETUNED_IMAGE_MODEL_PATH",
    str(BASE_DIR / "models" / "fashion-clip-finetuned"),
)

if USE_MODEL_TYPE == MODEL_TYPE_BASELINE:
    IMAGE_EMBEDDING_MODEL = BASELINE_MODEL_NAME
    MODEL_DISPLAY_NAME = "Baseline CLIP (ViT-B-32)"
elif USE_MODEL_TYPE == MODEL_TYPE_FINETUNED:
    IMAGE_EMBEDDING_MODEL = FINETUNED_MODEL_PATH
    MODEL_DISPLAY_NAME = "Fine-tuned Fashion CLIP"
else:
    raise ValueError(
        f"Invalid USE_MODEL_TYPE: {USE_MODEL_TYPE}. "
        f"Must be '{MODEL_TYPE_BASELINE}' or '{MODEL_TYPE_FINETUNED}'"
    )

IMAGE_VECTOR_SIZE = _env_int("IMAGE_VECTOR_SIZE", 512)


# --- LLM MODELS ---
# Keep Groq model names configurable to tune Vietnamese quality/latency.
LLM_MODEL_NAME = _env("LLM_MODEL_NAME", "llama-3.3-70b-versatile")
LLM_REWRITE_MODEL = _env("LLM_REWRITE_MODEL", "llama-3.1-8b-instant")


# --- SENTIMENT / VOICE ---
SENTIMENT_MODEL_NAME = _env(
    "SENTIMENT_MODEL_NAME",
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
)
VOICE_MODEL_SIZE = _env("VOICE_MODEL_SIZE", "small")
VOICE_LANGUAGE = _env("VOICE_LANGUAGE", "vi")


# --- COLLECTION NAMES ---
COLLECTION_PRODUCT_TEXT = _env("COLLECTION_PRODUCT_TEXT", "fashion_products")
COLLECTION_PRODUCT_IMAGE = _env("COLLECTION_PRODUCT_IMAGE", "fashion_images")
COLLECTION_POLICIES = _env("COLLECTION_POLICIES", "fashion_policies")


# --- SETTINGS ---
SEARCH_THRESHOLD = _env_float("SEARCH_THRESHOLD", 0.65)