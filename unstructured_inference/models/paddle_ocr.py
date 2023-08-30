paddle_ocr = None  # type: ignore


def load_agent():
    """Loads the PaddleOCR agent as a global variable to ensure that we only load it once."""

    from unstructured_paddleocr import PaddleOCR

    global paddle_ocr
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", mkl_dnn=True, show_log=False)

    return paddle_ocr
