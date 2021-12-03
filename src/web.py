import tempfile
from pathlib import Path
from typing import cast

import cv2
import torch.cuda
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from model.model_wrapper import LGPMAModel

app = FastAPI()
endpoint_extract_table_structure = "/extract_table_structure/"
endpoint_get_device = "/get_device/"
endpoint_set_device = "/set_device/{device}/"

model = LGPMAModel(device="cuda" if torch.cuda.is_available() else "cpu")


@app.get("/")
def return_success() -> JSONResponse:
    return JSONResponse({"status": "success"}, status_code=200)


@app.get(endpoint_get_device, response_class=JSONResponse)
def get_device() -> JSONResponse:
    return JSONResponse({"device": model.get_device()}, status_code=200)


@app.get(endpoint_set_device, response_class=JSONResponse)
def set_device(device: str):
    try:
        model.set_device(device)
        return JSONResponse(
            {"status": "success", "current_device": model.get_device()}, status_code=200
        )
    except Exception as e:
        return JSONResponse(
            {
                "status": "failure",
                "error": str(e),
                "current_device": model.get_device(),
            },
            status_code=200,
        )


@app.post(endpoint_extract_table_structure, response_class=JSONResponse)
async def extract_document_layout(
    input_file: UploadFile = File(...),
):
    extension = input_file.filename.split(".")[1]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = Path(tmpdir) / f"tmp.{extension}"
        with tmp_file.open("wb+") as f:
            f.write(cast(bytes, await input_file.read()))

        try:
            img = cv2.imread(str(tmp_file))
            result = model(img)
            return JSONResponse(result, status_code=200)

        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)
