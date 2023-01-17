from fastapi import FastAPI, File, status, Request, UploadFile, Form, HTTPException
from unstructured_inference.inference.layout import process_data_with_model
from unstructured_inference.models import UnknownModelException
from typing import List, Union
import tempfile
#from unstructured_inference.inference.layout import DocumentLayout
from unstructured_inference.layout_model import local_inference
from unstructured_inference.models import get_model
app = FastAPI()

ALL_ELEMS = "_ALL"
VALID_FILETYPES = ["pdf", "image"]


@app.post("/layout/{filetype:path}")
async def layout_parsing(
    filetype: str,
    file: UploadFile = File(),
    include_elems: List[str] = Form(default=ALL_ELEMS),
    model: str = Form(default=None),
):
    if filetype not in VALID_FILETYPES:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    is_image = filetype == "image"
    try:
        layout = process_data_with_model(file.file, model, is_image)
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


@app.post("/layout/v0.2/image")
async def layout_v02_parsing_image(
    request: Request,
    files: Union[List[UploadFile], None] = File(default=None),
):

    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_file.write(files[0].file.read())
        detections = local_inference(tmp_file.name, type="image", to_json=True)

    return detections


@app.post("/layout/v0.2/pdf")
async def layout_v02_parsing_pdf(
    request: Request,
    files: Union[List[UploadFile], None] = File(default=None),
):

    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_file.write(files[0].file.read())
        detections = local_inference(tmp_file.name, type="pdf", to_json=True)

    return detections

@app.get("/healthcheck", status_code=status.HTTP_200_OK)
async def healthcheck(request: Request):
    return {"healthcheck": "HEALTHCHECK STATUS: EVERYTHING OK!"}
