from typing import Tuple

import cv2
import numpy as np

from modules.models.base import Model
from modules.utils.image import pad_resize_image


class OpenCVModel(Model):

    __slots__ = ["face_net", "input_size",
                 "det_thres", "bbox_area_thres", "mean_values"]

    def __init__(
            self,
            face_net: cv2.dnn.Net,
            input_size: Tuple[int, int],
            det_thres: float,
            bbox_area_thres: float,
            FACE_MEAN_VALUES: Tuple[float, float, float] = (104.0, 117.0, 123.0)):
        Model.__init__(self, input_size, det_thres, bbox_area_thres)
        self.face_net = face_net
        self.FACE_MEAN_VALUES = FACE_MEAN_VALUES

    def __call__(
            self,
            cv2_img: np.ndarray):
        resized = pad_resize_image(cv2_img, new_size=self.input_size)
        # opencv expects BGR format
        blob = cv2.dnn.blobFromImage(
            resized, 1.0, self.input_size, self.FACE_MEAN_VALUES)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()[0][0]
        # reorder dets to have [xmin, ymin, xmax, ymax, conf] format
        # from a [_, _, conf, xmin, ymin, xmax, ymax] fmt
        return detections[:, [3, 4, 5, 6, 2]]


def batch_inference_img(opencv_model: OpenCVModel, cv2_img: np.ndarray):
    """Reference func for batched OpenCV DNN inference"""
    image = cv2_img
    blob = cv2.dnn.blobFromImages(
        image, 1.0, (300, 300), (104.0, 117.0, 123.0))
    opencv_model.face_net.setInput(blob)
    detections = opencv_model.face_net.forward()

    count = 0
    threshold = 0.5
    img_idx = 0
    (h, w) = image[0].shape[:2]

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability)
        # associated with the prediction
        confidence = detections[0, 0, i, 2]
        det_img_idx = int(detections[0, 0, i, 0])

        # filter weak detections
        if confidence > threshold:
            count += 1
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image[det_img_idx], (startX, startY),
                          (endX, endY), (0, 0, 255), 2)
            cv2.putText(image[det_img_idx], text, (startX, y),
                        cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 0, 255), 2)
        if (i + 1) % 200 == 0:
            cv2.imshow("output", image[img_idx])
            cv2.waitKey(0)
            img_idx += 1
