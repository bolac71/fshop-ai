import os
import shutil
from fastapi import UploadFile
from faster_whisper import WhisperModel
from app.services.image_service import ImageService
from app.core.config import VOICE_MODEL_SIZE, VOICE_LANGUAGE

class VoiceService:
    def __init__(self):
        print("Loading Whisper Model (Auto Speech Recognition)...")
        self.model_size = VOICE_MODEL_SIZE
        self.language = VOICE_LANGUAGE
        self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        print("Voice Service Ready!")

    def transcribe(self, file_path: str) -> str:
        """Chuyển file audio thành text"""
        segments, info = self.model.transcribe(
            file_path,
            beam_size=5,
            language=self.language,
            task="transcribe",
        )
        
        # Whisper trả về generator, cần loop để lấy text
        text_result = " ".join([segment.text for segment in segments])
        
        print(f"Detected language '{info.language}' with probability {info.language_probability}")
        return text_result.strip()

    async def process_voice_search(self, file: UploadFile, image_service: ImageService):
        """
        Pipeline: Audio -> Text -> Image Vector -> Search Result
        """
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            transcribed_text = self.transcribe(temp_filename)
            print(f"User said: {transcribed_text}")

            if not transcribed_text:
                return {"text": "", "results": []}

            points = image_service.search_by_text_vector(transcribed_text)
            
            return {
                "text": transcribed_text,
                "results": points
            }

        except Exception as e:
            print(f"Voice Search Error: {e}")
            return {"text": "Error processing audio", "results": []}
            
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)




  



  