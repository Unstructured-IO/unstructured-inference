paddle_ocr = None  # type: ignore


def load_agent():
    """Loads the PaddleOCR agent as a global variable to ensure that we only load it once."""

    from unstructured_paddleocr import PaddleOCR
    import paddle
    # Use paddlepaddle-gpu if there is gpu device available
    gpu_available  = (paddle.device.cuda.device_count() > 0)

    global paddle_ocr
    paddle_ocr = PaddleOCR(use_angle_cls=True, use_gpu=gpu_available, lang="en", mkl_dnn=True, show_log=False)

    return paddle_ocr
