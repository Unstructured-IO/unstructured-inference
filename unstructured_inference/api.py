from fastapi import FastAPI, File, status, Request, UploadFile, Form, HTTPException
from unstructured_inference.inference.layout import process_data_with_model
from unstructured_inference.models import UnknownModelException
from typing import List

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


@app.get("/healthcheck", status_code=status.HTTP_200_OK)
async def healthcheck(request: Request):
    return {"healthcheck": "HEALTHCHECK STATUS: EVERYTHING OK!"}
