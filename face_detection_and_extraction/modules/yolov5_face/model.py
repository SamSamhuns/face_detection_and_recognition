from typing import Any, Callable, Tuple

import numpy as np
from modules.models.base import Model


class YOLOV5FaceModel(Model):

    __slots__ = ["net", "inf_func"]

    def __init__(
            self,
            net: Any,
            det_thres: float,
            bbox_area_thres: float,
            inf_func: Callable,
            input_size: Tuple[int, int]):
        Model.__init__(self, input_size, det_thres, bbox_area_thres)

        self.net = net
        self.inf_func = inf_func

    def __call__(self,
                 cv2_img: np.ndarray) -> np.ndarray:
        iw, ih = self.input_size
        detections = self.inf_func(
            self.net, cv2_img, self.input_size).cpu().numpy()

        detections[:, :4] = detections[:, :4] / np.array([iw, ih, iw, ih])
        # reorder dets to have [xmin, ymin, xmax, ymax, conf, landmarks...] fmt
        return detections[:, :5]
