from pathlib import Path
from typing import Any, Union, Optional, List, Dict

import torch
from PIL import Image as PILImage

# from unstructured_inference.config import inference_config
from unstructured_inference.models.lore.ctdet import CtdetDetector
from unstructured_inference.models.lore.opts import opts
# # from unstructured_inference.models.unstructuredmodel import UnstructuredModel
# from unstructured_inference.logger import logger
# from unstructured_inference.utils import pad_image_with_background_color


# class UnstructuredLOREModel(UnstructuredModel):
#     def initialize(
#         self,
#         model: Union[str, Path] = None,
#         device: Optional[str] = "cuda" if torch.cuda.is_available() else "cpu",
#     ):
#         """Loads the donut model using the specified parameters"""
#         self.device = device
#
#         try:
#             logger.info("Loading the table structure model ...")
#             # self.model = LORE.from_pretrained(model)
#             self.model.eval()
#
#         except EnvironmentError:
#             logger.critical("Failed to initialize the model.")
#             logger.critical("Ensure that the model is correct")
#             raise ImportError(
#                 "Review the parameters to initialize a UnstructuredTableTransformerModel obj",
#             )
#         self.model.to(device)
#
#     def predict(
#         self,
#         x: PILImage.Image,
#         ocr_tokens: Optional[List[Dict]] = None,
#         result_format: str = "html",
#     ):
#         """Predict table structure deferring to run_prediction with ocr tokens
#
#         Note:
#         `ocr_tokens` is a list of dictionaries representing OCR tokens,
#         where each dictionary has the following format:
#         {
#             "bbox": [int, int, int, int],  # Bounding box coordinates of the token
#             "block_num": int,  # Block number
#             "line_num": int,   # Line number
#             "span_num": int,   # Span number
#             "text": str,  # Text content of the token
#         }
#         The bounding box coordinates should match the table structure.
#         FIXME: refactor token data into a dataclass so we have clear expectations of the fields
#         """
#         super().predict(x)
#         return self.run_prediction(x, ocr_tokens=ocr_tokens, result_format=result_format)
#
#     def run_prediction(
#         self,
#         x: PILImage.Image,
#         pad_for_structure_detection: int = inference_config.TABLE_IMAGE_BACKGROUND_PAD,
#         ocr_tokens: Optional[List[Dict]] = None,
#         result_format: Optional[str] = "html",
#     ):
#         """Predict table structure"""
#         outputs_structure = self.get_structure(x, pad_for_structure_detection)
#         if ocr_tokens is None:
#             raise ValueError("Cannot predict table structure with no OCR tokens")
#
#         # recognized_table = recognize(outputs_structure, x, tokens=ocr_tokens)
#         recognized_table = []
#         if len(recognized_table) > 0:
#             prediction = recognized_table[0]
#         # NOTE(robinson) - This means that the table was not recognized
#         else:
#             return ""
#
#         if result_format == "html":
#             pass
#             # Convert cells to HTML
#             # prediction = cells_to_html(prediction) or ""
#         elif result_format == "cells":
#             prediction = prediction
#         else:
#             raise ValueError(
#                 f"result_format {result_format} is not a valid format. "
#                 f'Valid formats are: "html", "dataframe", "cells"'
#             )
#
#         return prediction
#
#     def get_structure(
#         self,
#         x: PILImage.Image,
#         pad_for_structure_detection: int = inference_config.TABLE_IMAGE_BACKGROUND_PAD,
#     ) -> dict:
#         """get the table structure as a dictionary contaning different types of elements as
#         key-value pairs; check table-transformer documentation for more information"""
#         with torch.no_grad():
#             logger.info(f"padding image by {pad_for_structure_detection} for structure detection")
#             encoding = self.feature_extractor(
#                 pad_image_with_background_color(x, pad_for_structure_detection),
#                 return_tensors="pt",
#             ).to(self.device)
#             outputs_structure = self.model(**encoding)
#             outputs_structure["pad_for_structure_detection"] = pad_for_structure_detection
#             return outputs_structure

if __name__ == '__main__':
    opt = opts().init()
    opt.load_model = '/home/kamil/git/core-product/upstream-unstructured-inference/unstructured_inference/models/lore/model_best.pth'
    opt.load_processor = '/home/kamil/git/core-product/upstream-unstructured-inference/unstructured_inference/models/lore/processor_best.pth'
    opt.debug = 1
    opt.output_dir = '/home/kamil/git/core-product/upstream-unstructured-inference/unstructured_inference/models/lore'
    opt.demo_name = '/demo'
    detector = CtdetDetector(opt=opt)
    ret = detector.run(opt=opt, image_or_path_or_tensor='/home/kamil/git/core-product/od_tables/23-BERKSHIRE/0.jpg')
    print(ret)