from typing import Dict

from layoutparser.ocr.tesseract_agent import TesseractAgent, is_pytesseract_available

from unstructured_inference.logger import logger

ocr_agents: Dict[str, TesseractAgent] = {}


def load_agent(languages: str = "eng"):
    """Loads the Tesseract OCR agent as a global variable to ensure that we only load it once.

    Parameters
    ----------
    languages
        The languages to use for the Tesseract agent. To use a langauge, you'll first need
        to isntall the appropriate Tesseract language pack.
    """
    global ocr_agents

    if not is_pytesseract_available():
        raise ImportError(
            "Failed to load Tesseract. Ensure that Tesseract is installed. Example command: \n"
            "    >>> sudo apt install -y tesseract-ocr",
        )

    if languages not in ocr_agents:
        logger.info(f"Loading the Tesseract OCR agent for {languages} ...")
        ocr_agents[languages] = TesseractAgent(languages=languages)
