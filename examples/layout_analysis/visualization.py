import os
import pathlib
import sys

from unstructured_inference.inference.elements import ImageTextRegion
from unstructured_inference.inference.layout import process_file_with_model
from unstructured_inference.utils import write_image

CUR_DIR = pathlib.Path(__file__).parent.resolve()


def run(f_path, scope):
    annotation_data_map = {
        "final": None,
        "extracted": {"layout": {"color": "green", "width": 2}},
        "inferred": {"inferred_layout": {"color": "blue", "width": 2}},
        "ocr": {"ocr_layout": {"color": "yellow", "width": 2}},
    }

    f_basename = os.path.splitext(os.path.basename(f_path))[0]
    output_dir_path = os.path.join(output_basedir_path, f_basename)
    os.makedirs(output_dir_path, exist_ok=True)

    doc = process_file_with_model(
        f_path,
        model_name=None,
        analysis=True,
    )

    for idx, page in enumerate(doc.pages):
        if scope == "image_only":
            embedded_image_elements = [
                el for el in page.layout if isinstance(el, ImageTextRegion)
            ]
            inferred_image_elements = [
                el for el in page.inferred_layout if el.type == "Figure"
            ]
            final_image_elements = [el for el in page.elements if el.type == "Image"]

            page.layout = embedded_image_elements
            page.inferred_layout = inferred_image_elements
            page.elements = final_image_elements

        for action_type, action_value in annotation_data_map.items():
            img = page.annotate(annotation_data=action_value)
            output_f_path = os.path.join(output_dir_path, f"{f_basename}_{idx+1}_{action_type}.jpg")
            write_image(img, output_f_path)

        print(f"page_num: {idx+1} - n_total_elements: {len(page.elements)} - n_extracted_elements: "
              f"{len(page.layout)} - n_inferred_elements: {len(page.inferred_layout)} - "
              f"n_ocr_elements: {len(page.ocr_layout)}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(
            "Please provide the path to the file name as the first argument and the scope as the "
            "second argument.",
        )
        sys.exit(1)

    if sys.argv[2] not in ["all", "image_only"]:
        print("Invalid scope")
        sys.exit(1)

    output_basedir_path = os.path.join(CUR_DIR, "output")
    os.makedirs(output_basedir_path, exist_ok=True)

    run(f_path=sys.argv[1], scope=sys.argv[2])
