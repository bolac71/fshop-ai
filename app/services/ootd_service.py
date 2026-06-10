import os
import tempfile
from pathlib import Path

from gradio_client import Client, handle_file

from app.core.config import HF_TOKEN, OOTD_SPACE_ID

SLOT_TO_OOTD: dict[str, str] = {
    "top": "Upper-body",
    "bottom": "Lower-body",
    "dress": "Dress",
}


class OotdService:
    def __init__(self):
        self._client: Client | None = None

    def _get_client(self) -> Client:
        if self._client is None:
            if HF_TOKEN.strip():
                os.environ.setdefault("HF_TOKEN", HF_TOKEN.strip())
            self._client = Client(OOTD_SPACE_ID)
        return self._client

    def tryon(
        self,
        person_bytes: bytes,
        garment_bytes: bytes,
        garment_category: str = "Upper-body",
        n_steps: int = 20,
        image_scale: float = 2.0,
        seed: int = -1,
    ) -> bytes:
        with tempfile.TemporaryDirectory() as tmp_dir:
            person_path = Path(tmp_dir) / "person.jpg"
            garment_path = Path(tmp_dir) / "garment.jpg"
            person_path.write_bytes(person_bytes)
            garment_path.write_bytes(garment_bytes)

            client = self._get_client()
            result = client.predict(
                vton_img=handle_file(str(person_path)),
                garm_img=handle_file(str(garment_path)),
                category=garment_category,
                n_samples=1,
                n_steps=n_steps,
                image_scale=image_scale,
                seed=seed,
                api_name="/process_dc",
            )
            # result can be a Dict { path, url, ... } or a plain string path
            result_path = result["path"] if isinstance(result, dict) else result
            return Path(result_path).read_bytes()
