from typing import Tuple

import cv2
import numpy as np

from modules.models.base import Model
from modules.utils.image import pad_resize_image


class OpenCVDnnModel(Model):

    __slots__ = ["face_net", "input_size", "det_thres", "bbox_area_thres", "mean_values"]

    def __init__(self,
                 face_net: cv2.dnn.Net,
                 input_size: Tuple[int, int],
                 det_thres: float,
                 bbox_area_thres: float,
                 FACE_MEAN_VALUES: Tuple[float, float, float] = (104.0, 117.0, 123.0)):
        super(Model, self).__init__(input_size, det_thres, bbox_area_thres)
        self.face_net = face_net
        self.FACE_MEAN_VALUES = FACE_MEAN_VALUES

    def __call__(self, cv2_img: np.ndarray):
        resized = pad_resize_image(cv2_img, new_size=self.input_size)
        # opencv expects BGR format
        blob = cv2.dnn.blobFromImage(resized, 1.0, self.input_size, self.FACE_MEAN_VALUES)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()[0][0]
        # reorder dets to have [xmin, ymin, xmax, ymax, conf] format
        detections[:, 4], detections[:, :4] = detections[:, 2], detections[:, 3:7]
        return detections[:, :7]
