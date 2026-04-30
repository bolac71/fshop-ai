import io
import time
from dataclasses import dataclass

from fastapi import HTTPException, UploadFile
from faster_whisper import WhisperModel

from app.core.config import (
    VOICE_BEAM_SIZE,
    VOICE_COMPUTE_TYPE,
    VOICE_DEVICE,
    VOICE_INITIAL_PROMPT,
    VOICE_LANGUAGE,
    VOICE_MODEL_SIZE,
    VOICE_VAD_FILTER,
)
from app.services.image_service import ImageService


@dataclass
class TranscriptionResult:
    text: str
    language: str = "unknown"
    language_probability: float = 0.0
    avg_logprob: float = 0.0
    no_speech_probability: float = 0.0
    confidence: float = 0.0


class VoiceService:
    MAX_AUDIO_BYTES = 10 * 1024 * 1024
    ACCEPTED_AUDIO_TYPES = {
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "audio/x-wav",
        "audio/mp4",
        "audio/m4a",
        "audio/aac",
        "audio/webm",
        "application/octet-stream",
    }

    def __init__(self):
        print("Loading Whisper Model (Speech Recognition)...")
        self.model_size = VOICE_MODEL_SIZE
        self.language = None if VOICE_LANGUAGE in {"", "auto", "none"} else VOICE_LANGUAGE
        self.beam_size = VOICE_BEAM_SIZE
        self.initial_prompt = VOICE_INITIAL_PROMPT
        self.vad_filter = VOICE_VAD_FILTER
        self.model = WhisperModel(
            self.model_size,
            device=VOICE_DEVICE,
            compute_type=VOICE_COMPUTE_TYPE,
        )
        print(
            "Voice Service Ready! "
            f"model={self.model_size}, language={self.language or 'auto'}, "
            f"beam_size={self.beam_size}, vad_filter={self.vad_filter}"
        )

    def _transcribe_audio(self, audio_source) -> TranscriptionResult:
        segments, info = self.model.transcribe(
            audio_source,
            beam_size=self.beam_size,
            language=self.language,
            task="transcribe",
            initial_prompt=self.initial_prompt,
            condition_on_previous_text=False,
            vad_filter=self.vad_filter,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 250,
            },
            no_speech_threshold=0.45,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
        )

        segment_list = list(segments)
        text_result = " ".join(segment.text.strip() for segment in segment_list if segment.text)
        language = info.language if info else "unknown"
        language_probability = info.language_probability if info else 0.0
        avg_logprob = (
            sum(float(getattr(segment, "avg_logprob", 0.0)) for segment in segment_list) / len(segment_list)
            if segment_list
            else -1.0
        )
        no_speech_probability = (
            max(float(getattr(segment, "no_speech_prob", 0.0)) for segment in segment_list)
            if segment_list
            else 1.0
        )
        confidence = self._estimate_confidence(language_probability, avg_logprob, no_speech_probability)

        print(
            f"Detected language '{language}' with probability {language_probability:.2f}; "
            f"asr_confidence={confidence:.2f}; no_speech={no_speech_probability:.2f}"
        )
        return TranscriptionResult(
            text=text_result.strip(),
            language=language,
            language_probability=float(language_probability or 0.0),
            avg_logprob=float(avg_logprob or 0.0),
            no_speech_probability=float(no_speech_probability or 0.0),
            confidence=confidence,
        )

    def _estimate_confidence(
        self,
        language_probability: float,
        avg_logprob: float,
        no_speech_probability: float,
    ) -> float:
        logprob_score = max(0.0, min(1.0, (avg_logprob + 1.5) / 1.5))
        speech_score = 1.0 - max(0.0, min(1.0, no_speech_probability))
        confidence = (0.45 * language_probability) + (0.35 * logprob_score) + (0.20 * speech_score)
        return round(max(0.0, min(1.0, confidence)), 4)

    def transcribe(self, file_path: str) -> str:
        return self._transcribe_audio(file_path).text

    def transcribe_from_buffer(self, audio_buffer: bytes) -> str:
        return self.transcribe_with_metadata_from_buffer(audio_buffer).text

    def transcribe_with_metadata_from_buffer(self, audio_buffer: bytes) -> TranscriptionResult:
        try:
            return self._transcribe_audio(io.BytesIO(audio_buffer))
        except Exception as e:
            print(f"Transcription error: {e}")
            raise

    async def transcribe_upload(self, file: UploadFile):
        """
        Lightweight pipeline: Audio -> ASR only. Clients can let users edit
        the transcript and then run the normal catalog search endpoint.
        """
        started_at = time.time()
        self._validate_upload(file)
        audio_buffer = await file.read()
        if not audio_buffer:
            raise HTTPException(status_code=400, detail="Audio file is empty")
        if len(audio_buffer) > self.MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Audio size exceeds maximum limit of 10MB")

        transcription = self.transcribe_with_metadata_from_buffer(audio_buffer)
        transcribed_text = transcription.text
        debug_reason = None
        if not transcribed_text or transcription.confidence < 0.25 or transcription.no_speech_probability >= 0.85:
            debug_reason = "empty_or_low_confidence_speech"

        return {
            "transcribed_text": transcribed_text,
            "normalized_query": transcribed_text,
            "rewritten_query": transcribed_text,
            "intent": "product_search",
            "filters": {},
            "asr_confidence": transcription.confidence,
            "asr": self._asr_payload(transcription),
            "latency_ms": int((time.time() - started_at) * 1000),
            "search_debug": {"reason": debug_reason} if debug_reason else None,
        }

    async def process_voice_search(self, file: UploadFile, image_service: ImageService, catalog_search_service=None):
        """
        Pipeline: Audio -> ASR -> Query understanding -> Catalog retrieval/rerank.
        """
        started_at = time.time()
        try:
            self._validate_upload(file)
            audio_buffer = await file.read()
            if not audio_buffer:
                raise HTTPException(status_code=400, detail="Audio file is empty")
            if len(audio_buffer) > self.MAX_AUDIO_BYTES:
                raise HTTPException(status_code=413, detail="Audio size exceeds maximum limit of 10MB")

            transcription = self.transcribe_with_metadata_from_buffer(audio_buffer)
            transcribed_text = transcription.text
            print(f"User said: {transcribed_text}")

            if not transcribed_text or transcription.confidence < 0.25 or transcription.no_speech_probability >= 0.85:
                return {
                    "transcribed_text": transcribed_text,
                    "text": transcribed_text,
                    "products": [],
                    "results": [],
                    "normalized_query": "",
                    "rewritten_query": "",
                    "intent": "product_search",
                    "filters": {},
                    "asr_confidence": transcription.confidence,
                    "asr": self._asr_payload(transcription),
                    "latency_ms": int((time.time() - started_at) * 1000),
                    "search_debug": {"reason": "empty_or_low_confidence_speech"},
                }

            if catalog_search_service is not None:
                catalog_result = catalog_search_service.search(transcribed_text, limit=12, debug=True)
                return {
                    "transcribed_text": transcribed_text,
                    "text": transcribed_text,
                    "products": catalog_result["products"],
                    "results": [],
                    "normalized_query": catalog_result.get("normalized_query", ""),
                    "rewritten_query": catalog_result.get("rewritten_query", ""),
                    "intent": catalog_result.get("intent", "product_search"),
                    "filters": catalog_result.get("filters", {}),
                    "asr_confidence": transcription.confidence,
                    "asr": self._asr_payload(transcription),
                    "latency_ms": int((time.time() - started_at) * 1000),
                    "search_debug": catalog_result.get("search_debug"),
                }

            points = image_service.search_by_text_vector(transcribed_text, limit=12)

            return {
                "transcribed_text": transcribed_text,
                "text": transcribed_text,
                "products": [],
                "results": points,
                "asr_confidence": transcription.confidence,
                "asr": self._asr_payload(transcription),
                "latency_ms": int((time.time() - started_at) * 1000),
                "search_debug": {"reason": "catalog_search_service_not_available"},
            }

        except HTTPException:
            raise
        except Exception as e:
            print(f"Voice Search Error: {e}")
            return {
                "transcribed_text": "",
                "text": "Error processing audio",
                "products": [],
                "results": [],
                "asr_confidence": 0.0,
                "latency_ms": int((time.time() - started_at) * 1000),
                "search_debug": {"reason": "exception", "error": str(e)},
            }

    def _validate_upload(self, file: UploadFile) -> None:
        content_type = (file.content_type or "application/octet-stream").lower()
        if content_type not in self.ACCEPTED_AUDIO_TYPES and not content_type.startswith("audio/"):
            raise HTTPException(status_code=415, detail=f"Unsupported audio content type: {content_type}")

    def _asr_payload(self, transcription: TranscriptionResult) -> dict:
        return {
            "language": transcription.language,
            "language_probability": transcription.language_probability,
            "avg_logprob": transcription.avg_logprob,
            "no_speech_probability": transcription.no_speech_probability,
            "confidence": transcription.confidence,
            "model": self.model_size,
            "device": VOICE_DEVICE,
            "compute_type": VOICE_COMPUTE_TYPE,
        }
