from io import BytesIO

from PIL import Image

from app.core.config import GEMINI_API_KEY, VTO_GEMINI_MODEL


class GeminiQuotaExceededError(RuntimeError):
    pass


class GeminiTryonService:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if not GEMINI_API_KEY.strip():
            raise ValueError("Missing GEMINI_API_KEY")
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=GEMINI_API_KEY.strip())
        return self._client

    def _normalize_image(self, image_bytes: bytes) -> tuple[bytes, str]:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        output = BytesIO()
        image.save(output, format="JPEG", quality=92)
        return output.getvalue(), "image/jpeg"

    def _build_prompt(self, user_prompt: str) -> str:
        base_prompt = (
            "Create one realistic ecommerce outfit preview image. "
            "Use the first image as the person/model reference and preserve the person's face, body shape, pose, "
            "camera angle, lighting, and background as much as possible. "
            "Use the second image as a contact sheet of garment/accessory references. "
            "Apply only the visible listed items from the contact sheet onto the person. "
            "Preserve each item's color, silhouette, logo, print, and material as closely as possible. "
            "Do not invent extra garments or accessories. "
            "Return only the final edited image."
        )
        user_prompt = (user_prompt or "").strip()
        return f"{base_prompt}\nAdditional styling request: {user_prompt}" if user_prompt else base_prompt

    def tryon_outfit(self, person_bytes: bytes, garment_sheet_bytes: bytes, prompt: str = "") -> bytes:
        from google.genai import types

        person_jpeg, person_mime = self._normalize_image(person_bytes)
        sheet_jpeg, sheet_mime = self._normalize_image(garment_sheet_bytes)
        request_prompt = self._build_prompt(prompt)
        client = self._get_client()

        last_text = ""
        for attempt in range(2):
            effective_prompt = request_prompt if attempt == 0 else (
                "Edit the first image of the person so they wear the clothing/accessory items shown in the second image. "
                "Keep the person identity and output only one realistic image."
            )
            try:
                response = client.models.generate_content(
                    model=VTO_GEMINI_MODEL,
                    contents=[
                        effective_prompt,
                        types.Part.from_bytes(data=person_jpeg, mime_type=person_mime),
                        types.Part.from_bytes(data=sheet_jpeg, mime_type=sheet_mime),
                    ],
                    config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
                )
            except Exception as exc:
                message = str(exc)
                if "429" in message or "RESOURCE_EXHAUSTED" in message or "Quota exceeded" in message:
                    raise GeminiQuotaExceededError(
                        "Gemini image quota/rate limit exceeded. Please check Google AI Studio billing/quota or try again later."
                    ) from exc
                raise

            candidates = getattr(response, "candidates", None) or []
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None) or []
                for part in parts:
                    text = getattr(part, "text", None)
                    if text:
                        last_text = text
                    inline_data = getattr(part, "inline_data", None)
                    if inline_data and getattr(inline_data, "data", None):
                        return inline_data.data

        detail = f" Gemini response: {last_text[:200]}" if last_text else ""
        raise RuntimeError(f"Gemini did not return an image.{detail}")
