import os

import pytest

from unstructured_inference.inference.layout import process_file_with_model


@pytest.mark.slow()
def test_layout_yolox_local_parsing_image():
    filename = os.path.join("sample-docs", "test-image.jpg")
    # NOTE(benjamin) keep_output = True create a file for each image in
    # localstorage for visualization of the result
    document_layout = process_file_with_model(filename, model_name="yolox", is_image=True)
    # NOTE(benjamin) The example image should result in one page result
    assert len(document_layout.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 13 detections
    types_known = ["Text","Section-header","Page-header"]
    known_regions=[e for e in document_layout.pages[0].elements if e.type in types_known]
    assert len(known_regions) == 13
    assert hasattr(
        document_layout.pages[0].elements[0],
        "prob",
    )  # NOTE(pravin) New Assertion to Make Sure LayoutElement has probabilities
    assert isinstance(
        document_layout.pages[0].elements[0].prob,
        float,
    )  # NOTE(pravin) New Assertion to Make Sure Populated Probability is Float


@pytest.mark.slow()
def test_layout_yolox_local_parsing_pdf():
    filename = os.path.join("sample-docs", "loremipsum.pdf")
    document_layout = process_file_with_model(filename, model_name="yolox")
    content = str(document_layout)
    assert "libero fringilla" in content
    assert len(document_layout.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 5 text detections
    text_elements=[e for e in document_layout.pages[0].elements if e.type=='Text']
    assert len(text_elements) == 5
    assert hasattr(
        document_layout.pages[0].elements[0],
        "prob",
    )  # NOTE(pravin) New Assertion to Make Sure LayoutElement has probabilities
    assert isinstance(
        document_layout.pages[0].elements[0].prob,
        float,
    )  # NOTE(pravin) New Assertion to Make Sure Populated Probability is Float


@pytest.mark.slow()
def test_layout_yolox_local_parsing_empty_pdf():
    filename = os.path.join("sample-docs", "empty-document.pdf")
    document_layout = process_file_with_model(filename, model_name="yolox")
    assert len(document_layout.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 0 detections
    assert len(document_layout.pages[0].elements) == 0


########################
# ONLY SHORT TESTS BELOW
########################


def test_layout_yolox_local_parsing_image_soft():
    filename = os.path.join("sample-docs", "test-image.jpg")
    # NOTE(benjamin) keep_output = True create a file for each image in
    # localstorage for visualization of the result
    document_layout = process_file_with_model(filename, model_name="yolox_tiny", is_image=True)
    # NOTE(benjamin) The example image should result in one page result
    assert len(document_layout.pages) == 1
    # NOTE(benjamin) Soft version of the test, run make test-long in order to run with full model
    assert len(document_layout.pages[0].elements) > 0
    assert hasattr(
        document_layout.pages[0].elements[0],
        "prob",
    )  # NOTE(pravin) New Assertion to Make Sure LayoutElement has probabilities
    assert isinstance(
        document_layout.pages[0].elements[0].prob,
        float,
    )  # NOTE(pravin) New Assertion to Make Sure Populated Probability is Float


def test_layout_yolox_local_parsing_pdf_soft():
    filename = os.path.join("sample-docs", "loremipsum.pdf")
    document_layout = process_file_with_model(filename, model_name="yolox_tiny")
    content = str(document_layout)
    assert "libero fringilla" in content
    assert len(document_layout.pages) == 1
    # NOTE(benjamin) Soft version of the test, run make test-long in order to run with full model
    assert len(document_layout.pages[0].elements) > 0
    assert hasattr(
        document_layout.pages[0].elements[0],
        "prob",
    )  # NOTE(pravin) New Assertion to Make Sure LayoutElement has probabilities
    assert (
        document_layout.pages[0].elements[0].prob is None
    )  # NOTE(pravin) New Assertion to Make Sure Uncategorized Text has None Probability


def test_layout_yolox_local_parsing_empty_pdf_soft():
    filename = os.path.join("sample-docs", "empty-document.pdf")
    document_layout = process_file_with_model(filename, model_name="yolox_tiny")
    assert len(document_layout.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 0 detections
    text_elements_page_1 = [el for el in document_layout.pages[0].elements if el.type != "Image"]
    assert len(text_elements_page_1) == 0
