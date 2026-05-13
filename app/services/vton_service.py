import os
import tempfile
from pathlib import Path

from gradio_client import Client, handle_file

from app.core.config import HF_TOKEN, VTON_SPACE_ID


class VtonService:
    def __init__(self):
        self._client: Client | None = None

    def _get_client(self) -> Client:
        if self._client is None:
            if HF_TOKEN.strip():
                os.environ.setdefault("HF_TOKEN", HF_TOKEN.strip())
            self._client = Client(VTON_SPACE_ID)
        return self._client

    def tryon(
        self,
        person_bytes: bytes,
        garment_bytes: bytes,
        garment_desc: str = "A nice shirt",
        denoise_steps: int = 30,
        seed: int = 42,
    ) -> bytes:
        with tempfile.TemporaryDirectory() as tmp_dir:
            person_path = Path(tmp_dir) / "person.jpg"
            garment_path = Path(tmp_dir) / "garment.jpg"
            person_path.write_bytes(person_bytes)
            garment_path.write_bytes(garment_bytes)

            client = self._get_client()

            result = client.predict(
                dict={
                    "background": handle_file(str(person_path)),
                    "layers": [],
                    "composite": None,
                },
                garm_img=handle_file(str(garment_path)),
                garment_des=garment_desc,
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=denoise_steps,
                seed=seed,
                api_name="/tryon",
            )

            # result is a tuple: (result_image_path, masked_image_path)
            result_image_path = result[0]
            return Path(result_image_path).read_bytes()
