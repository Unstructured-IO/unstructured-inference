import os
import pathlib
import sys

from unstructured_inference.inference.layout import process_file_with_model
from unstructured_inference.utils import write_image

CUR_DIR = pathlib.Path(__file__).parent.resolve()


def run(f_path):
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
        for action_type, action_value in annotation_data_map.items():
            img = page.annotate(annotation_data=action_value)
            output_f_path = os.path.join(output_dir_path, f"{f_basename}_{idx+1}_{action_type}.jpg")
            write_image(img, output_f_path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(
            "Please provide the path to the file name as the first argument and the strategy as the "
            "second argument.",
        )
        sys.exit(1)

    output_basedir_path = os.path.join(CUR_DIR, "output")
    os.makedirs(output_basedir_path, exist_ok=True)

    run(f_path=sys.argv[1])
