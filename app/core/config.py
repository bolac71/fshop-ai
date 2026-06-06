import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

# Local monorepo convenience: the NestJS backend owns the canonical DB config.
# Load it as a secondary env source so the AI service does not silently query a
# different local database during order lookup.
BACKEND_ENV_PATH = PROJECT_ROOT.parent / "fshop-be" / ".env"
if BACKEND_ENV_PATH.exists():
    load_dotenv(BACKEND_ENV_PATH, override=False)


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
    "dbname": _env("DATABASE_NAME", _env("DB_NAME", "fshop_db")),
    "user": _env("DATABASE_USER", _env("DB_USER", "username")),
    "password": _env("DATABASE_PASSWORD", _env("DB_PASSWORD", "123456")),
    "host": _env("DATABASE_HOST", _env("DB_HOST", "localhost")),
    "port": _env("DATABASE_PORT", _env("DB_PORT", "5432")),
}

if _env("DB_USE_SSL", "false").strip().lower() in {"1", "true", "yes", "on"}:
    POSTGRES_CONFIG["sslmode"] = "require"


# --- QDRANT CONFIG ---
QDRANT_URL = _env("QDRANT_URL", "http://localhost:6333")


# --- REDIS CONFIG ---
REDIS_URL = _env("REDIS_URL", "redis://localhost:6379")


# --- MODEL NAMES ---
# Vietnamese SBERT: Optimized for Vietnamese semantic understanding
# Better than multilingual for Vietnamese domain-specific queries
TEXT_EMBEDDING_MODEL = _env(
    "TEXT_EMBEDDING_MODEL",
    "keepitreal/vietnamese-sbert-base",
)
TEXT_VECTOR_SIZE = _env_int("TEXT_VECTOR_SIZE", 768)
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


# --- MODERATION ---
# Set to your HuggingFace Hub repo ID after training (e.g. "username/fshop-phobert-moderation")
PHOBERT_MODERATION_MODEL = _env("PHOBERT_MODERATION_MODEL", "")
MODERATION_RULE_WEIGHT = _env_float("MODERATION_RULE_WEIGHT", 0.30)
MODERATION_FLAG_THRESHOLD = _env_float("MODERATION_FLAG_THRESHOLD", 0.35)
MODERATION_HIGH_PRIORITY_THRESHOLD = _env_float("MODERATION_HIGH_PRIORITY_THRESHOLD", 0.70)


# --- VOICE ---
VOICE_MODEL_SIZE = _env("VOICE_MODEL_SIZE", "medium")
VOICE_LANGUAGE = _env("VOICE_LANGUAGE", "vi").strip().lower()
VOICE_DEVICE = _env("VOICE_DEVICE", "cpu")
VOICE_COMPUTE_TYPE = _env("VOICE_COMPUTE_TYPE", "int8")
VOICE_BEAM_SIZE = _env_int("VOICE_BEAM_SIZE", 5)
VOICE_VAD_FILTER = _env("VOICE_VAD_FILTER", "true").strip().lower() in {"1", "true", "yes", "on"}
VOICE_INITIAL_PROMPT = _env(
    "VOICE_INITIAL_PROMPT",
    (
        "Tim kiem san pham thoi trang bang tieng Viet. "
        "Vi du: toi muon mua vay, ao so mi, quan jean, ao khoac, "
        "giay, tui xach, mau trang, mau den, nam, nu."
    ),
)


# --- COLLECTION NAMES ---
COLLECTION_PRODUCT_TEXT = _env("COLLECTION_PRODUCT_TEXT", "fashion_products")
COLLECTION_PRODUCT_IMAGE = _env("COLLECTION_PRODUCT_IMAGE", "fashion_images")
COLLECTION_POLICIES = _env("COLLECTION_POLICIES", "fashion_policies")


# --- VIRTUAL TRY-ON ---
VTON_SPACE_ID = _env("VTON_SPACE_ID", "hysts-duplicates/IDM-VTON")
HF_TOKEN = _env("HF_TOKEN", "")


# --- SETTINGS ---
# Cosine similarity thresholds for vector search:
# - Image search: 0.65 (high precision, reduce false positives)
# - Text search (voice): 0.62 (optimized for Vietnamese SBERT)
# - Personalized search: 0.6 (user profile based)
SEARCH_THRESHOLD = _env_float("SEARCH_THRESHOLD", 0.65)
