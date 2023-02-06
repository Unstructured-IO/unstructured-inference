from fastapi import FastAPI, File, status, Request, UploadFile, Form, HTTPException
from unstructured_inference.inference.layout import process_data_with_model
from unstructured_inference.models.base import UnknownModelException
from typing import List
import tempfile

from unstructured_inference.models.yolox import yolox_local_inference

app = FastAPI()

ALL_ELEMS = "_ALL"
VALID_FILETYPES = ["pdf", "image"]


@app.post("/layout/detectron/{filetype:path}")
async def layout_parsing(
    filetype: str,
    file: UploadFile = File(default=None),
    include_elems: List[str] = Form(default=ALL_ELEMS),
    model: str = Form(default=None),
):
    if filetype not in VALID_FILETYPES:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    is_image = filetype == "image"
    try:
        layout = process_data_with_model(file.file, model, is_image)  # type: ignore
    except UnknownModelException as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))
    pages_layout = [
        {
            "number": page.number,
            "elements": [
                element.to_dict()
                for element in page.elements
                if element.type in include_elems or include_elems == ALL_ELEMS
            ],
        }
        for page in layout.pages
    ]

    return {"pages": pages_layout}


@app.post("/layout/yolox/{filetype:path}")
async def layout_parsing_yolox(
    filetype: str,
    request: Request,
    file: List[UploadFile] = File(default=None),
    force_ocr=Form(default=False),
):
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_file.write(file[0].file.read())
        detections = yolox_local_inference(tmp_file.name, type=filetype, use_ocr=force_ocr)

    return detections


@app.get("/healthcheck", status_code=status.HTTP_200_OK)
async def healthcheck(request: Request):
    return {"healthcheck": "HEALTHCHECK STATUS: EVERYTHING OK!"}
