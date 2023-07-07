import json
import os
import tempfile
from datetime import datetime

import pdf2image

from engine import run_ocr_with_layout_detection
from unstructured_inference.models.base import get_model
from unstructured_inference.models.unstructuredmodel import UnstructuredObjectDetectionModel, \
    UnstructuredElementExtractionModel


def run(filename, page_limit=None, drawable=True):
    print(">>> Start", filename)

    now_dt = datetime.utcnow()
    now_str = now_dt.strftime("%Y_%m_%d-%H_%M_%S")

    f_path = os.path.join(example_docs_dir, filename)

    sub_output_dir = os.path.join(output_dir, now_str)
    if drawable:
        os.makedirs(sub_output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        images = pdf2image.convert_from_path(f_path, output_folder=tmpdir)

    individual_page_img_paths = []
    individual_page_images = []
    for i, image in enumerate(images):
        # Save the image to a file
        # img_path = os.path.join(sub_output_dir, f"page_{i+1}.jpg")
        # image.save(img_path)
        # individual_page_img_paths.append(img_path)

        individual_page_images.append(image)

    n_pages = len(individual_page_images)
    print(f"number_of_pages: {n_pages}")

    page_limit = page_limit if page_limit else n_pages

    print("individual_page_images:")
    for i, image in enumerate(individual_page_images[:page_limit]):
        print(f"\timage{i + 1} - size: {image.size}")

    page_size = individual_page_images[0].size if len(individual_page_images) > 0 else None

    model = get_model()
    if isinstance(model, UnstructuredObjectDetectionModel):
        detection_model = model
        element_extraction_model = None
        model_type = "UnstructuredObjectDetectionModel"
    elif isinstance(model, UnstructuredElementExtractionModel):
        detection_model = None
        element_extraction_model = model
        model_type = "UnstructuredElementExtractionModel"
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    print("model_type:", model_type)

    # OCR'ing individual blocks
    print("OCR'ing individual blocks...")

    individual_page_images = individual_page_images[:page_limit]

    infer_time_individual, text_individual_dict = run_ocr_with_layout_detection(
        images=individual_page_images,
        detection_model=detection_model,
        element_extraction_model=element_extraction_model,
        mode="individual_blocks",
        output_dir=sub_output_dir,
        drawable=drawable,
    )

    # OCR'ing entire page
    print("OCR'ing entire page...")

    infer_time_entire, text_entire_dict = run_ocr_with_layout_detection(
        images=individual_page_images,
        detection_model=detection_model,
        element_extraction_model=element_extraction_model,
        mode="entire_page",
        output_dir=sub_output_dir,
        drawable=drawable,
    )

    print("Processing Time (OCR'ing individual blocks)")
    print(f"\ttotal_infer_time: {infer_time_individual}")
    print(f"\tavg_infer_time_per_page: {infer_time_individual / n_pages}")

    print("Processing Time (OCR'ing entire page)")
    print(f"\ttotal_infer_time: {infer_time_entire}")
    print(f"\tavg_infer_time_per_page: {infer_time_entire / n_pages}")

    delimiter = " "

    text_individual = delimiter.join(text_individual_dict.values())
    text_entire = delimiter.join(text_entire_dict.values())

    # Calculate similarity ratio
    from difflib import SequenceMatcher
    similarity_ratio = SequenceMatcher(None, text_individual, text_entire).ratio()

    print(f"similarity_ratio: {similarity_ratio}")

    import nltk

    # Download the required resources (run this once)
    nltk.download('punkt')

    # Tokenize the text into words
    word_list_individual = nltk.word_tokenize(text_individual)
    n_word_list_individual = len(word_list_individual)
    print("n_word_list_in_text_individual:", n_word_list_individual)
    word_sets_individual = set(list(word_list_individual))
    n_word_sets_individual = len(word_sets_individual)
    print(f"n_word_sets_in_text_individual: {n_word_sets_individual}")
    # print("word_sets_merged:", word_sets_merged)

    word_list_entire = nltk.word_tokenize(text_entire)
    n_word_list_entire = len(word_list_entire)
    print("n_word_list_individual:", n_word_list_entire)
    word_sets_entire = set(list(word_list_entire))
    n_word_sets_entire = len(word_sets_entire)
    print(f"n_word_sets_individual: {n_word_sets_entire}")
    # print("word_sets_individual:", word_sets_individual)

    # Find unique elements using difference
    print("diff_elements:")
    unique_words_individual = word_sets_individual - word_sets_entire
    unique_words_entire = word_sets_entire - word_sets_individual
    print(f"unique_words_in_text_individual: {unique_words_individual}\n")
    print(f"unique_words_in_text_entire: {unique_words_entire}")

    report = {
        "file_info": {
            "filename": filename,
            "n_pages": n_pages,
            "page_size": page_size,
        },
        "model": {
            "model": str(model),
            "model_type": model_type,
        },
        "processing_time": {
            "individual_blocks": infer_time_individual,
            "entire_page": infer_time_entire,
        },
        "text_similarity": {
            "similarity_ratio": similarity_ratio,
            "individual_blocks": {
                "n_word_list": n_word_list_individual,
                "n_word_sets": n_word_sets_individual,
                "unique_words": delimiter.join(list(unique_words_individual)),
            },
            "entire_page": {
                "n_word_list": n_word_list_entire,
                "n_word_sets": n_word_sets_entire,
                "unique_words": delimiter.join(list(unique_words_entire)),
            }
        },
        "extracted_text": {
            "individual_blocks": {
                "text_by_page": text_individual_dict,
                "text_all": text_individual,
            },
            "entire_page": {
                "text_by_page": text_entire_dict,
                "text_all": text_entire,
            },
        }
    }

    report_f_name = f"validate-ocr-{now_str}.json"
    report_f_path = os.path.join(output_dir, report_f_name)
    with open(report_f_path, "w", encoding="utf-8-sig") as f:
        json.dump(report, f, indent=4)


if __name__ == '__main__':
    cur_dir = os.getcwd()
    base_dir = os.path.join(cur_dir, os.pardir, os.pardir)
    example_docs_dir = os.path.join(base_dir, "sample-docs")

    # folder path to save temporary outputs
    output_dir = os.path.join(cur_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    filenames = [
        "2023-Jan-economic-outlook.pdf",
        # "recalibrating-risk-report.pdf",
        # "Silent-Giant.pdf",
        # "loremipsum_multipage.pdf",
        # "layout-parser-paper.pdf",
    ]

    for f_name in filenames:
        run(f_name, None, True)
