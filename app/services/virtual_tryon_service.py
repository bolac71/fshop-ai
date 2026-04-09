import io
import base64
from PIL import Image
from typing import Optional
import os
from dotenv import load_dotenv
from gradio_client import Client, handle_file
import tempfile
import requests

load_dotenv()

class VirtualTryonService:
    """
    Virtual Try-on Service using Hugging Face Gradio Spaces (FREE)
    Using IDM-VTON model from yisol/IDM-VTON space
    """
    
    def __init__(self):
        # Gradio space URL for IDM-VTON (FREE to use)
        self.space_url = "yisol/IDM-VTON"
        self.client = None
        self.hf_token = os.getenv("HF_TOKEN")  # Token will be auto-detected by gradio_client
        
        try:
            print(f"🔄 Connecting to Hugging Face Space: {self.space_url}...")
            if self.hf_token:
                print(f"🔐 HF_TOKEN detected - will authenticate automatically")
            else:
                print(f"⚠️  No HF_TOKEN found, connecting as anonymous (limited quota)")
            
            # Gradio Client automatically uses HF_TOKEN from environment
            self.client = Client(self.space_url)
            print(f"✅ Virtual Try-on Service initialized successfully")
            print(f"   Space: {self.space_url}")
            print(f"   Auth: {'Authenticated' if self.hf_token else 'Anonymous'}")
        except Exception as e:
            print(f"⚠️ Warning: Could not connect to Gradio space: {e}")
            print(f"   Service will attempt to reconnect on first request")
    
    def _ensure_client(self):
        """Ensure client is connected, reconnect if needed"""
        if self.client is None:
            try:
                print(f"🔄 Reconnecting to {self.space_url}...")
                if self.hf_token:
                    print(f"🔐 Using authenticated connection (HF_TOKEN detected)...")
                # Gradio Client automatically uses HF_TOKEN from environment
                self.client = Client(self.space_url)
                print(f"✅ Connected successfully")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Gradio space: {e}")

    def _image_to_base64(self, image_data: bytes) -> str:
        """Convert image bytes to base64 string"""
        return base64.b64encode(image_data).decode('utf-8')

    def _base64_to_image(self, base64_string: str) -> Image.Image:
        """Convert base64 string back to PIL Image"""
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))

    async def tryon_garment(
        self,
        person_image: bytes,
        garment_image: bytes,
        category: str = "upper_body",
        garment_des: str = "",
        seed: int = 42,
        steps: int = 30,
        crop: bool = False,
        force_dc: bool = False,
        mask_only: bool = False,
    ) -> dict:
        """
        Apply virtual try-on effect using Gradio IDM-VTON (FREE).
        
        Args:
            person_image: Bytes of the person image
            garment_image: Bytes of the garment/clothing image
            category: "upper_body", "lower_body", "dresses" (note: dresses not dress)
            garment_des: Description of the garment (optional)
            seed: Random seed (default: 42)
            steps: Inference steps (default: 30)
            crop: Auto-crop (default: False)
            force_dc: Force DC (default: False)
            mask_only: Return only mask (default: False)
            
        Returns:
            dict with success, image, base64, message
        """
        try:
            self._ensure_client()
            
            # Save images to temp files (Gradio requires file paths)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as person_f:
                person_f.write(person_image)
                person_path = person_f.name
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as garment_f:
                garment_f.write(garment_image)
                garment_path = garment_f.name
            
            try:
                print(f"🔄 Sending virtual try-on request to Gradio Space...")
                print(f"   Category: {category}, Steps: {steps}, Seed: {seed}")
                if garment_des:
                    print(f"   Description: {garment_des}")
                
                # Call Gradio API - IDM-VTON parameters
                # API signature: predict(dict, garm_img, garment_des, is_checked, is_checked_crop, denoise_steps, seed)
                result = self.client.predict(
                    {"background": handle_file(person_path), "layers": [], "composite": None},  # dict_param
                    handle_file(garment_path),  # garm_img
                    garment_des,  # garment_des
                    True,  # is_checked (use auto-generated mask)
                    crop,  # is_checked_crop
                    steps,  # denoise_steps
                    seed,  # seed
                    api_name="/tryon"
                )
                
                # Result is tuple: (result_image_path, masked_image_path)
                if not result or not result[0]:
                    raise ValueError("No result from Gradio API")
                
                result_image_path = result[0]
                print(f"📥 Result path: {result_image_path}")
                
                # Check if result is a URL or local path
                if isinstance(result_image_path, str) and (
                    result_image_path.startswith('http://') or 
                    result_image_path.startswith('https://')
                ):
                    # Download from URL
                    print(f"🌐 Downloading result from URL...")
                    response = requests.get(result_image_path, timeout=30)
                    response.raise_for_status()
                    result_image = Image.open(io.BytesIO(response.content))
                else:
                    # Read from local file
                    print(f"📁 Reading result from local file...")
                    result_image = Image.open(result_image_path)
                
                # Convert to base64
                buffer = io.BytesIO()
                result_image.save(buffer, format="PNG")
                result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                print(f"✅ Virtual try-on completed successfully!")
                print(f"   Output image size: {result_image.size}")
                
                return {
                    "success": True,
                    "image": result_image,
                    "base64": result_base64,
                    "message": "Virtual try-on completed successfully (FREE via Gradio)"
                }
                
            finally:
                # Cleanup temp files
                try:
                    os.unlink(person_path)
                    os.unlink(garment_path)
                except:
                    pass
            
        except Exception as e:
            error_msg = f"Gradio API error: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "image": None,
                "base64": None,
                "message": error_msg
            }

    async def tryon_batch(
        self,
        person_image: bytes,
        garments: list,
    ) -> list:
        """
        Apply virtual try-on with multiple garments for the same person.
        
        Args:
            person_image: Bytes of the person image
            garments: List of dicts with keys:
                - image: bytes of garment image
                - category: str (upper_body, lower_body, dress)
                - garment_des: str (optional)
                - name: str (optional, for tracking)
                - seed: int (optional)
                - steps: int (optional)
                
        Returns:
            List of result dicts
        """
        results = []
        
        for idx, garment in enumerate(garments):
            garment_image = garment.get("image")
            category = garment.get("category", "upper_body")
            garment_des = garment.get("garment_des", "")
            seed = garment.get("seed", 42 + idx)  # Vary seed for different results
            steps = garment.get("steps", 30)
            name = garment.get("name", f"garment_{idx}")
            
            result = await self.tryon_garment(
                person_image=person_image,
                garment_image=garment_image,
                category=category,
                garment_des=garment_des,
                seed=seed,
                steps=steps,
            )
            
            result["name"] = name
            result["category"] = category
            results.append(result)
        
        return results

    async def get_supported_garment_types(self) -> list:
        """
        Get list of supported garment categories for IDM-VTON
        """
        return ["upper_body", "lower_body", "dress"]

    def resize_image_for_tryon(
        self,
        image_data: bytes,
        max_dimension: int = 768
    ) -> bytes:
        """
        Resize image for optimal virtual try-on processing.
        Recommended size: 512-768px for best results.
        
        Args:
            image_data: Original image bytes
            max_dimension: Maximum width/height in pixels
            
        Returns:
            Resized image as bytes
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Resize while maintaining aspect ratio
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Save to bytes
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=95)
            
            return output.getvalue()
        except Exception as e:
            print(f"❌ Error resizing image: {str(e)}")
            raise

    def crop_image_for_tryon(
        self,
        image_data: bytes,
        crop_box: Optional[tuple] = None
    ) -> bytes:
        """
        Crop image for virtual try-on. Crop box format: (left, top, right, bottom)
        Useful to focus on specific areas.
        
        Args:
            image_data: Original image bytes
            crop_box: Tuple of (left, top, right, bottom) or None for auto-crop
            
        Returns:
            Cropped image as bytes
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            
            if crop_box:
                image = image.crop(crop_box)
            
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Save to bytes
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=95)
            
            return output.getvalue()
        except Exception as e:
            print(f"❌ Error cropping image: {str(e)}")
            raise
