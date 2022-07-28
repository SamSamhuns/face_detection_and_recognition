from typing import Tuple

import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN

from modules.models.base import Model


class MTCNNSlowModel(Model):

    __slots__ = ["mtcnn"]

    def __init__(
            self,
            det_thres: float,
            bbox_area_thres: float):
        Model.__init__(self, (None, None), det_thres, bbox_area_thres)
        self.mtcnn = MTCNN()

    def __call__(
            self,
            cv2_img: np.ndarray) -> np.ndarray:
        model_out = self.mtcnn.detect_faces(cv2_img)
        # set model input_size to input image size
        self.input_size = cv2_img.shape[:2][::-1]
        iw, ih = self.input_size

        detections = []
        # reorder dets to have [xmin, ymin, xmax, ymax, lmarks, conf] format
        for det in model_out:
            x, y, width, height = det['box']
            xmin, ymin, xmax, ymax = x, y, x + width, y + height
            xmin, ymin, xmax, ymax = xmin / iw, ymin / ih, xmax / iw, ymax / ih
            conf = det['confidence']
            lmarks = np.asarray([kp for kp in det['keypoints'].values()]).flatten()
            lmarks = np.asarray([(kp0 / iw, kp1 / ih) for kp0, kp1 in det['keypoints'].values()]).flatten()
            detections.append([xmin, ymin, xmax, ymax, *lmarks, conf])
        detections = np.empty(shape=(0, 15)) if not detections else detections
        return np.asarray(detections)


class MTCNNFastModel(Model):

    __slots__ = ["mtcnn_func", "min_size", "factor", "thresholds"]

    def __init__(
            self,
            model_path: str,
            det_thres: float,
            bbox_area_thres: float,
            min_size: int = 40,
            factor: float = 0.7,
            thresholds: Tuple[int, int, int] = (0.6, 0.7, 0.8),
            device: str = "cpu:0"):
        Model.__init__(self, (None, None), det_thres, bbox_area_thres)

        self.min_size = min_size
        self.factor = factor
        self.thresholds = thresholds

        def _mtcnn_func(img, min_size, factor, thresholds):
            with open(model_path, "rb") as f:
                graph_def = tf.compat.v1.GraphDef.FromString(f.read())

            with tf.device(f"/{device}"):
                prob, landmarks, box = tf.compat.v1.import_graph_def(
                    graph_def,
                    input_map={
                        "input:0": img,
                        "min_size:0": min_size,
                        "factor:0": factor,
                        "thresholds:0": thresholds
                    },
                    return_elements=[
                        "prob:0",
                        "landmarks:0",
                        "box:0"], name='')
            return box, prob, landmarks

        # wrap graph function as a callable function
        self.mtcnn_func = tf.compat.v1.wrap_function(_mtcnn_func, [
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
            tf.TensorSpec(shape=[3], dtype=tf.float32)
        ])

    def __call__(
            self,
            cv2_img: np.ndarray) -> np.ndarray:
        # set model input_size to input image size
        self.input_size = cv2_img.shape[:2][::-1]
        iw, ih = self.input_size

        bboxes, scores, landmarks = self.mtcnn_func(cv2_img, self.min_size, self.factor, self.thresholds)
        bboxes, scores, landmarks = bboxes.numpy(), scores.numpy(), landmarks.numpy()
        # bboxes and landmarks must be normalized to input_size
        # reorder bboxes to have [xmin,ymin,xmax,ymax] fmt from [ymin,xmin,ymax,xmax]
        bboxes = bboxes[:, [1, 0, 3, 2]] / np.array([iw, ih, iw, ih])
        # reorder landmarks to have [x1,y1,x2,y2,x3,y3,x4,y4,x5,y5] fmt from [y1,y2,y3,y4,y5,x1,x2,x3,x4,x5]
        landmarks = landmarks[:, [5, 0, 6, 1, 7, 2, 8, 3, 9, 4]] / np.array([iw, ih, iw, ih, iw, ih, iw, ih, iw, ih])

        return np.concatenate([bboxes, landmarks, np.expand_dims(scores, axis=1)], axis=1)
