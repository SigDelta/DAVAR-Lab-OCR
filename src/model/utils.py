from pathlib import Path
from typing import Dict

import cv2
import numpy as np


def visualize_result(img: np.ndarray, result: Dict, output_path: Path) -> None:
    bboxes = [
        [b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]] for b in result["bboxes"]
    ]
    for box in bboxes:
        for j in range(0, len(box), 2):
            cv2.line(
                img,
                (box[j], box[j + 1]),
                (box[(j + 2) % len(box)], box[(j + 3) % len(box)]),
                (0, 0, 255),
                1,
            )
    output_path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(output_path), img)
