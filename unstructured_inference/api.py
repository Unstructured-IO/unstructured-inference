from fastapi import FastAPI, File, status, Request, UploadFile, Form, HTTPException
from unstructured_inference.inference.layout import process_data_with_model
from unstructured_inference.models import UnknownModelException
from typing import List, BinaryIO, Optional, Union

app = FastAPI()

ALL_ELEMS = "_ALL"


@app.post("/layout/pdf")
async def layout_parsing_pdf(
    file: UploadFile = File(),
    include_elems: List[str] = Form(default=ALL_ELEMS),
    model: str = Form(default=None),
):
    return get_pages_layout(file.file, model, include_elems)


@app.post("/layout/image")
async def layout_parsing_image(
    file: UploadFile = File(),
    include_elems: List[str] = Form(default=ALL_ELEMS),
    model: str = Form(default=None),
):
    return get_pages_layout(file.file, model, include_elems, is_image=True)


def get_pages_layout(
    file: BinaryIO,
    model: Optional[str],
    include_elems: Union[List[str], str] = ALL_ELEMS,
    is_image=False,
):
    try:
        layout = process_data_with_model(file, model, is_image)
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
