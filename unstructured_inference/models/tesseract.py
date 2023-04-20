from layoutparser.ocr.tesseract_agent import is_pytesseract_available, TesseractAgent

from unstructured_inference.logger import logger

ocr_agent: TesseractAgent = None


def load_agent(languages: str = "eng"):
    """Loads the Tesseract OCR agent as a global variable to ensure that we only load it once.

    Parameters
    ----------
    languages
        The languages to use for the Tesseract agent. To use a langauge, you'll first need
        to isntall the appropriate Tesseract language pack.
    """
    global ocr_agent

    if not is_pytesseract_available():
        raise ImportError(
            "Failed to load Tesseract. Ensure that Tesseract is installed. Example command: \n"
            "    >>> sudo apt install -y tesseract-ocr"
        )

    if ocr_agent is None:
        logger.info("Loading the Tesseract OCR agent ...")
        ocr_agent = TesseractAgent(languages=languages)
