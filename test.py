import os
import time

from PIL import Image
from tqdm import tqdm

import unstructured_inference.models.detectron2onnx as detectron2
from unstructured_inference.inference.layout import DocumentLayout
from unstructured_inference.logger import logger, logger_onnx


def annotate_with_model(model,prefix,suffix,output_folder,file_path):
    
    doc = DocumentLayout.from_file(filename=file_path,
                                    detection_model=model)
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for i,p in enumerate(doc.pages):
        im = p.annotate()
        im.save(f"{output_folder}/{prefix}-{i}-{suffix}.png")


model_old = detectron2.UnstructuredDetectronONNXModel()
model_old.initialize(**detectron2.MODEL_TYPES['detectron2_mask_rcnn'])

model_new = detectron2.UnstructuredDetectronONNXModel()
model_new.initialize(**detectron2.MODEL_TYPES['detectron2_onnx'])


files = [
        "sample-docs/layout-parser-paper.pdf",
        "../unstructured/example-docs/fake-memo.pdf",
        "../unstructured/example-docs/reliance.pdf",
        ]

for i,file in tqdm(enumerate(files)):
    
    t1 = time.monotonic()
    annotate_with_model(model_old,f"{i}","old","results",file)
    t2 = time.monotonic()

    print(f"Old model took {t2-t1} seconds to process")
    t3 = time.monotonic()
    annotate_with_model(model_new,f"{i}","new","results",file)
    t4 = time.monotonic()
    print(f"New model took {t4-t3} seconds")
