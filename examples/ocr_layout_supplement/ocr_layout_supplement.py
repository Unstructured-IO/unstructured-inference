import os
import pathlib
import sys

from unstructured_inference.constants import AnnotationResult
from unstructured_inference.inference.layout import process_file_with_model
from unstructured_inference.utils import annotate_layout_elements

CUR_DIR = pathlib.Path(__file__).parent.resolve()


def run(f_path, file_type):
    print(">>> Start...")
    print(f">>> file_path: {f_path} - file_type: {file_type}")

    if file_type == "pdf":
        is_image = False
    elif file_type == "image":
        is_image = True
    else:
        print("Invalid file type.")
        sys.exit(1)

    annotation_data_map = {
        "final": None,
    }

    actions = [False, True]
    for action in actions:
        _f_basename = os.path.splitext(os.path.basename(f_path))[0]
        output_dir_path = os.path.join(output_basedir_path, f"{_f_basename}_{file_type}")
        os.makedirs(output_dir_path, exist_ok=True)

        f_basename = f"updated_{_f_basename}" if action else f"original_{_f_basename}"

        doc = process_file_with_model(
            f_path,
            is_image=is_image,
            model_name=None,
            supplement_with_ocr_elements=action,
            analysis=True,
        )

        annotate_layout_elements(doc, annotation_data_map, output_dir_path, f_basename, AnnotationResult.IMAGE)

    print("<<< Finished")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(
            "Please provide the path to the file name as the first argument and the strategy as the "
            "second argument.",
        )
        sys.exit(1)

    output_basedir_path = os.path.join(CUR_DIR, "output")
    os.makedirs(output_basedir_path, exist_ok=True)

    run(f_path=sys.argv[1], file_type=sys.argv[2])
