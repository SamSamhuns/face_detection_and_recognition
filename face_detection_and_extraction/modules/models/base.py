# base Model and Detections classes
import numpy as np
from typing import Tuple, Optional, List, Any


class Model:

    __slots__ = ["input_size", "det_thres", "bbox_area_thres", "returns_opt_labels"]

    def __init__(
            self,
            input_size: Tuple[int, int],
            det_thres: float,
            bbox_area_thres: float,
            returns_opt_labels: bool = False):
        """
        Model class that runs the face detection inference
        Args:
            input_size: input shape (width, height)
            det_thres: detection threshold
            bbox_area_thres: detection bbox area threshold
            returns_opt_labels: flag to state if class instance returns optional labels
                                or a Tuple after being called
        """
        self.input_size = input_size
        self.det_thres = det_thres
        self.bbox_area_thres = bbox_area_thres
        self.returns_opt_labels = returns_opt_labels

    def __call__(self):
        raise NotImplementedError("__call__ method has not been implemented")


class PostProcessedDetection:

    __slots__ = ["boxes", "bbox_confs", "bbox_areas", "bbox_lmarks", "bbox_labels"]

    def __init__(
            self,
            boxes: np.ndarray,
            bbox_confs: np.ndarray,
            bbox_areas: np.ndarray,
            bbox_lmarks: Optional[np.ndarray] = None,
            bbox_labels: Optional[List[Any]] = None):
        """
        Stores the post-processed detection results that can be used for drawing on orig input image
        Args:
            boxes: Bounding box coords, 2D np.ndarray [[xmin, ymin, xmax, ymax], ...]
            bbox_confs: Bbox conf scores, np.ndarray [float, float, ...]
            bbox_areas: Bbox areas as percentages of the total input image area [float, float, ...]
            bbox_lmarks: Optional bbox face landmark coords in pairs, 2D np.ndarray [[float11, float12, float21, float22, ...]]
            bbox_labels: Optional bbox labels with info such as face age, gender, etc
        """
        self.boxes = boxes
        self.bbox_confs = bbox_confs
        self.bbox_areas = bbox_areas
        self.bbox_lmarks = bbox_lmarks
        self.bbox_labels = bbox_labels
