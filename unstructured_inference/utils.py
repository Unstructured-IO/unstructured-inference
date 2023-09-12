import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterator, Union

import cv2
import numpy as np
from PIL import Image

from unstructured_inference.constants import AnnotationResult
from unstructured_inference.visualize import show_plot

if TYPE_CHECKING:
    from unstructured_inference.inference.layout import DocumentLayout


class LazyEvaluateInfo:
    """Class that stores the information needed to lazily evaluate a function with given arguments.
    The object stores the information needed for evaluation as a function and its arguments.
    """

    def __init__(self, evaluate: Callable, *args, **kwargs):
        self.evaluate = evaluate
        self.info = (args, kwargs)


class LazyDict(Mapping):
    """Class that wraps a dict and only evaluates keys of the dict when the key is accessed. Keys
    that should be evaluated lazily should use LazyEvaluateInfo objects as values. By default when
    a value is computed from a LazyEvaluateInfo object, it is converted to the raw value in the
    internal dict, so subsequent accessing of the key will produce the same value. Set cache=False
    to avoid storing the raw value.
    """

    def __init__(self, *args, cache=True, **kwargs):
        self.cache = cache
        self._raw_dict = dict(*args, **kwargs)

    def __getitem__(self, key: Hashable) -> Union[LazyEvaluateInfo, Any]:
        value = self._raw_dict.__getitem__(key)
        if isinstance(value, LazyEvaluateInfo):
            evaluate = value.evaluate
            args, kwargs = value.info
            value = evaluate(*args, **kwargs)
            if self.cache:
                self._raw_dict[key] = value
        return value

    def __iter__(self) -> Iterator:
        return iter(self._raw_dict)

    def __len__(self) -> int:
        return len(self._raw_dict)


def write_image(image: Union[Image.Image, np.ndarray], output_image_path: str):
    """
    Write an image to a specified file path, supporting both PIL Image and numpy ndarray formats.

    Parameters:
    - image (Union[Image.Image, np.ndarray]): The image to be written, which can be in PIL Image
      format or a numpy ndarray format.
    - output_image_path (str): The path to which the image will be written.

    Raises:
    - ValueError: If the provided image type is neither PIL Image nor numpy ndarray.

    Returns:
    - None: The function writes the image to the specified path but does not return any value.
    """

    if isinstance(image, Image.Image):
        image.save(output_image_path)
    elif isinstance(image, np.ndarray):
        cv2.imwrite(output_image_path, image)
    else:
        raise ValueError("Unsupported Image Type")


def annotate_layout_elements(
    doc: "DocumentLayout",
    annotation_data_map: dict,
    output_dir_path: str,
    output_f_basename: str,
    result: AnnotationResult = AnnotationResult.IMAGE,
    plot_desired_width: int = 14,
):
    """
    Annotates layout elements on each page of the document and saves or displays the result.

    This function iterates through each page of the given document and applies annotations based on
    the given action type and action value in the annotation_data_map. The annotated images are then
    either saved to the disk or displayed as plots.

    Parameters:
    - doc (DocumentLayout): The document layout object containing the pages to annotate.
    - annotation_data_map (dict): A mapping from action types to action values defining the
                                  annotations to be applied.
    - output_dir_path (str): The directory path where the annotated images will be saved.
    - output_f_basename (str): The base name to use for the output image files.
    - result (str, optional): Specifies the result type. Can be either
                              'ANNOTATION_RESULT_WITH_IMAGE' for saving the annotated images
                              or 'ANNOTATION_RESULT_WITH_PLOT' for displaying them as plots.
                              Default is 'ANNOTATION_RESULT_WITH_IMAGE'.
    - plot_desired_width (int, optional): The desired width for the plot when result is set to
                                          'ANNOTATION_RESULT_WITH_PLOT'. Default is 14.

    Note:
    - If the 'result' parameter is set to 'ANNOTATION_RESULT_WITH_IMAGE', the annotated images will
      be saved in the directory specified by 'output_dir_path'.
    - If the 'result' parameter is set to 'ANNOTATION_RESULT_WITH_PLOT', the annotated images will
      be displayed as plots and not saved.
    """

    for idx, page in enumerate(doc.pages):
        for action_type, action_value in annotation_data_map.items():
            img = page.annotate(annotation_data=action_value)
            output_f_path = os.path.join(
                output_dir_path,
                f"{output_f_basename}_{idx + 1}_{action_type}.jpg",
            )
            if result == AnnotationResult.IMAGE:
                write_image(img, output_f_path)
                print(f"wrote {output_f_path}")
            elif result == AnnotationResult.PLOT:
                show_plot(img, desired_width=plot_desired_width)


def pad_image_with_background_color(
    image: Image.Image,
    pad: int = 10,
    background_color: str = "white",
) -> Image.Image:
    """pads an input image with the same background color around it by pad//2 on all 4 sides

    The original image is kept intact and a new image is returned with padding added.
    """
    width, height = image.size
    if pad < 0:
        raise ValueError(
            "Can not pad an image with negative space! Please use a positive value for `pad`.",
        )
    new = Image.new(image.mode, (width + pad, height + pad), background_color)
    new.paste(image, (pad // 2, pad // 2))
    return new
