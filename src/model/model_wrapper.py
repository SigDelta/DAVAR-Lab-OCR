from typing import Dict
from xml.dom import minidom

import numpy as np

from davarocr import inference_model, init_model


class LGPMAModel:
    def __init__(
        self,
        device: str = "cpu",
        checkpoint_path: str = "models/lgpma/maskrcnn-lgpma-pub-e12-pub.pth",
        config_path: str = "src/configs/lgpma_pub.py",
    ):
        self._checkpoint_path = checkpoint_path
        self._config_path = config_path
        self._device = device
        self._model = None

    def get_device(self) -> str:
        return self._device

    def set_device(self, device: str):
        self._model = init_model(
            self._config_path, self._checkpoint_path, device=device
        )
        self._device = device
        print(f"Using device: {device}.")

    def __call__(self, img: np.ndarray) -> Dict:
        if self._model is None:
            self.set_device(self._device)

        try:
            result = inference_model(self._model, img)[0]
            result["status"] = "success"
            result["html"] = minidom.parseString(result["html"]).toprettyxml()

        except Exception as e:
            result = {"status": "failure", "error": str(e)}

        return result
