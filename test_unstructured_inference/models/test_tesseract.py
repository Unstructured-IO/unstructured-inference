# from unittest.mock import patch

# import pytest

# from unstructured_inference.models import tesseract


# class MockTesseractAgent:
#     def __init__(self, languages):
#         pass


# def test_load_agent(monkeypatch):
#     monkeypatch.setattr(tesseract, "TesseractAgent", MockTesseractAgent)
#     monkeypatch.setattr(tesseract, "ocr_agents", {})

#     with patch.object(tesseract, "is_pytesseract_available", return_value=True):
#         tesseract.load_agent(languages="eng+swe")

#     assert isinstance(tesseract.ocr_agents["eng+swe"], MockTesseractAgent)


# def test_load_agent_raises_when_not_available():
#     with patch.object(tesseract, "is_pytesseract_available", return_value=False):
#         with pytest.raises(ImportError):
#             tesseract.load_agent()
