import numpy as np
from mtcnn.mtcnn import MTCNN

from modules.models.base import Model


class MTCNNModel(Model):

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
        # reorder dets to have [xmin, ymin, xmax, ymax, conf, lmarks] format
        for det in model_out:
            x, y, width, height = det['box']
            xmin, ymin, xmax, ymax = x, y, x + width, y + height
            xmin, ymin, xmax, ymax = xmin / iw, ymin / ih, xmax / iw, ymax / ih
            conf = det['confidence']
            lmarks = np.asarray([kp for kp in det['keypoints'].values()]).flatten()
            detections.append([xmin, ymin, xmax, ymax, conf, *lmarks])
        detections = np.empty(shape=(0, 15)) if not detections else detections
        return np.asarray(detections)
