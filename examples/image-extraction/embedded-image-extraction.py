import io
import os.path
import pathlib
import sys

import fitz  # PyMuPDF
from PIL import Image
from PyPDF2 import PdfReader

from unstructured_inference.inference.layout import DocumentLayout

CUR_DIR = pathlib.Path(__file__).parent.resolve()


def print_result(images, page_index):
    if images:
        print(f"[+] Found a total of {len(images)} images in page {page_index}")
    else:
        print(f"[!] No images found on page {page_index}")


def run_with_unstructured(f_path, output_dir_path):
    doc = DocumentLayout.from_file(
        filename=f_path,
        extract_images_in_pdf=True,
        image_output_dir_path=output_dir_path,
    )

    for page_index, page in enumerate(doc.pages, start=1):
        image_elements = [el for el in page.elements if el.type == "Image"]
        print_result(image_elements, page_index)


def run_with_pymupdf(f_path, output_dir_path):
    doc = fitz.open(f_path)
    for page_index, page in enumerate(doc, start=1):
        image_list = page.get_images(full=True)
        print_result(image_list, page_index)

        for image_index, img in enumerate(image_list, start=1):
            # Get the XREF of the image
            xref = img[0]
            # Extract the image bytes
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            # Get the image extension
            image_ext = base_image["ext"]
            # Load it to PIL
            image = Image.open(io.BytesIO(image_bytes))
            output_f_path = os.path.join(output_dir_path, f"image_{page_index}_{image_index}.{image_ext}")
            image.save(output_f_path)


def run_with_pypdf2(f_path, output_dir_path):
    reader = PdfReader(f_path)
    for page_index, page in enumerate(reader.pages, start=1):
        images = page.images
        print_result(images, page_index)

        for image_file_object in images:
            output_f_path = os.path.join(output_dir_path, f"figure_{page_index}_{image_file_object.name}")
            with open(output_f_path, "wb") as fp:
                fp.write(image_file_object.data)


def run(f_path, library):
    output_dir_path = os.path.join(output_basedir_path, library)
    os.makedirs(output_dir_path, exist_ok=True)

    if library == "unstructured":
        run_with_unstructured(f_path, output_dir_path)
    elif library == "pymupdf":
        run_with_pymupdf(f_path, output_dir_path)
    elif library == "pypdf2":
        run_with_pypdf2(f_path, output_dir_path)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(
            "Please provide the path to the file name as the first argument and the image "
            "extraction library as the second argument.",
        )
        sys.exit(1)

    if sys.argv[2] not in ["unstructured", "pymupdf", "pypdf2"]:
        print("Invalid pdf library")
        sys.exit(1)

    output_basedir_path = os.path.join(CUR_DIR, "output")
    os.makedirs(output_basedir_path, exist_ok=True)

    run(f_path=sys.argv[1], library=sys.argv[2])
