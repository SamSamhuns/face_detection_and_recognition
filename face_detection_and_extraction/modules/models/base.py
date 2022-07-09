# base Model and Detections classes
import numpy as np
from typing import Tuple, Optional


class Model:

    def __init__(
            self,
            input_size: Tuple[int, int],
            det_thres: float,
            bbox_area_thres: float):
        """Model class that runs the face inference"""
        self.input_size = input_size
        self.det_thres = det_thres
        self.bbox_area_thres = bbox_area_thres

    def __call__(self):
        raise NotImplementedError("__call__ method has not been implemented")


class PostProcessedDetection:

    __slots__ = ["boxes", "bbox_confs", "bbox_areas", "bbox_lmarks"]

    def __init__(
            self,
            boxes: np.ndarray,
            bbox_confs: np.ndarray,
            bbox_areas: np.ndarray,
            bbox_lmarks: Optional[np.ndarray] = None):
        """Stores the post-processed detection results that can be used for drawing on orig input image"""
        self.boxes = boxes
        self.bbox_confs = bbox_confs
        self.bbox_areas = bbox_areas
        self.bbox_lmarks = bbox_lmarks
