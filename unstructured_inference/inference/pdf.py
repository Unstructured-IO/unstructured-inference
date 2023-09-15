from typing import List, Any

from pdfminer.layout import LTContainer, LTImage


def get_images(layout_object: Any) -> List[LTImage]:
    # recursively locate Image objects in layout_object
    if isinstance(layout_object, LTImage):
        return [layout_object]
    if isinstance(layout_object, LTContainer):
        img_list = []
        for child in layout_object:
            img_list = img_list + get_images(child)
        return img_list
    else:
        return []
