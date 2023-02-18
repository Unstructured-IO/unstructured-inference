import os

from unstructured_inference.inference.layout import process_file_with_model


def test_layout_yolox_local_parsing_image():
    filename = os.path.join("sample-docs", "test-image.jpg")
    # NOTE(benjamin) keep_output = True create a file for each image in
    # localstorage for visualization of the result
    document_layout = process_file_with_model(filename, model_name="yolox", is_image=True)
    # NOTE(benjamin) The example image should result in one page result
    assert len(document_layout.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 13 detections
    assert len(document_layout.pages[0].elements) == 13


def test_layout_yolox_local_parsing_pdf():
    filename = os.path.join("sample-docs", "loremipsum.pdf")
    document_layout = process_file_with_model(filename, model_name="yolox")
    content = str(document_layout)
    assert "Lorem ipsum" in content
    assert len(document_layout.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 5 detections
    assert len(document_layout.pages[0].elements) == 5


def test_layout_yolox_local_parsing_empty_pdf():
    filename = os.path.join("sample-docs", "empty-document.pdf")
    document_layout = process_file_with_model(filename, model_name="yolox")
    assert len(document_layout.pages) == 1
    # NOTE(benjamin) The example sent to the test contains 0 detections
    assert len(document_layout.pages[0].elements) == 0
