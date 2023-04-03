from fastapi import FastAPI, File, status, Request, UploadFile, Form, HTTPException
from unstructured_inference.inference.layout import process_data_with_model
from unstructured_inference.models.base import UnknownModelException
from typing import List

app = FastAPI()

ALL_ELEMS = "_ALL"
VALID_FILETYPES = ["pdf", "image"]


@app.post("/layout/{modeltype:path}/{filetype:path}")
async def layout_parsing(
    filetype: str,
    modeltype: str,
    file: UploadFile = File(default=None),
    include_elems: List[str] = Form(default=ALL_ELEMS),
    force_ocr=Form(default=False),
    # TODO(alan): Need a way to send model options to the model
):
    """Route to proper filetype parser."""
    if filetype not in VALID_FILETYPES:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    is_image = filetype == "image"
    model = None if modeltype == "default" else modeltype
    ocr_strategy = "force" if force_ocr else "auto"
    try:
        layout = process_data_with_model(
            file.file, model, is_image, ocr_strategy=ocr_strategy
        )  # type: ignore
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


@app.get("/healthcheck", status_code=status.HTTP_200_OK)
async def healthcheck(request: Request):
    """Return healthy status"""
    return {"healthcheck": "HEALTHCHECK STATUS: EVERYTHING OK!"}
