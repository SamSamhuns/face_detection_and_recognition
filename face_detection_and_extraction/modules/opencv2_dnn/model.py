from typing import Tuple, List

import cv2
import numpy as np

from modules.models.base import Model
from modules.utils.image import scale_coords
from modules.utils.image import pad_resize_image


class OpenCVFaceDetModel(Model):

    __slots__ = ["face_net", "face_mean_values"]

    def __init__(
            self,
            face_net: cv2.dnn.Net,
            input_size: Tuple[int, int],
            det_thres: float,
            bbox_area_thres: float,
            FACE_MEAN_VALUES: Tuple[float, float, float] = (104.0, 117.0, 123.0)):
        Model.__init__(self, input_size, det_thres, bbox_area_thres)
        self.face_net = face_net
        self.face_mean_values = FACE_MEAN_VALUES

    def __call__(
            self,
            cv2_img: np.ndarray) -> np.ndarray:
        resized = pad_resize_image(cv2_img, new_size=self.input_size)
        # opencv expects BGR format
        blob = cv2.dnn.blobFromImage(
            resized, 1.0, self.input_size, self.face_mean_values)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()[0][0]
        # reorder dets to have [xmin, ymin, xmax, ymax, conf] format
        # from a [_, _, conf, xmin, ymin, xmax, ymax] fmt
        return detections[:, [3, 4, 5, 6, 2]]


class OpenCVFaceAgeModel(Model):
    __slots__ = ["age_net", "age_mean_values"]

    def __init__(
            self,
            age_net: cv2.dnn.Net,
            det_thres: float,
            bbox_area_thres: float,
            INPUT_SIZE: Tuple[int, int] = (227, 227),
            AGE_MEAN_VALUES: Tuple[float, float, float] = (78.4263377603, 87.7689143744, 114.895847746)):
        """
        Predicts age groups ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        """
        Model.__init__(self, INPUT_SIZE, det_thres, bbox_area_thres)
        self.age_net = age_net
        self.age_mean_values = AGE_MEAN_VALUES

    def __call__(
            self,
            cv2_img: np.ndarray,
            pad_resize: bool = False) -> np.ndarray:
        if pad_resize:
            cv2_img = pad_resize_image(cv2_img, new_size=self.input_size)
        # opencv expects BGR format
        blob = cv2.dnn.blobFromImage(
            cv2_img, 1.0, self.input_size, self.age_mean_values, swapRB=False)
        self.age_net.setInput(blob)
        preds = self.age_net.forward()[0]
        # preds contains logits for 8 age group preds, run argmax to get most confident pred
        # ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        return preds


class OpenCVFaceGenderModel(Model):
    __slots__ = ["gender_net", "gender_mean_values"]

    def __init__(
            self,
            gender_net: cv2.dnn.Net,
            det_thres: float,
            bbox_area_thres: float,
            INPUT_SIZE: Tuple[int, int] = (227, 227),
            GENDER_MEAN_VALUES: Tuple[float, float, float] = (78.4263377603, 87.7689143744, 114.895847746)):
        """
        Predicts two genders ["Male", "Female"]
        """
        Model.__init__(self, INPUT_SIZE, det_thres, bbox_area_thres)
        self.gender_net = gender_net
        self.gender_mean_values = GENDER_MEAN_VALUES

    def __call__(
            self,
            cv2_img: np.ndarray,
            pad_resize: bool = False) -> np.ndarray:
        if pad_resize:
            cv2_img = pad_resize_image(cv2_img, new_size=self.input_size)
        # opencv expects BGR format
        blob = cv2.dnn.blobFromImage(
            cv2_img, 1.0, self.input_size, self.gender_mean_values, swapRB=False)
        self.gender_net.setInput(blob)
        preds = self.gender_net.forward()[0]
        # preds contains logits for 2 gender preds, run argmax to get most confident pred
        # ['Male', 'Female']
        return preds


class OpenCVFaceDetAgeGenderModel(Model):

    __slots__ = ["face_net", "age_net",
                 "gender_net", "age_list", "gender_list"]

    def __init__(
            self,
            face_net: cv2.dnn.Net,
            age_net: cv2.dnn.Net,
            gender_net: cv2.dnn.Net,
            input_size: Tuple[int, int],
            det_thres: float,
            bbox_area_thres: float):
        Model.__init__(self, input_size, det_thres, bbox_area_thres, returns_opt_labels=True)
        self.face_net = OpenCVFaceDetModel(
            face_net, input_size, det_thres, bbox_area_thres)
        self.age_net = OpenCVFaceAgeModel(
            age_net, det_thres, bbox_area_thres)
        self.gender_net = OpenCVFaceGenderModel(
            gender_net, det_thres, bbox_area_thres)
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                         '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']

    def __call__(
            self,
            cv2_img: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Returns a tuple of face dets and age_gender pred txt labels
        """
        h, w = cv2_img.shape[:2]
        mw, mh = self.input_size
        face_dets = self.face_net(cv2_img)
        opt_labels = []

        # filter face dets below threshold
        face_dets = face_dets[face_dets[:, 4] > self.det_thres]
        # scale coords to orig image size
        bboxes = face_dets[:, :4] * np.array([mw, mh, mw, mh])
        bboxes = scale_coords((mh, mw), bboxes, (h, w)).round()

        padding = 5
        for bbox in bboxes:
            bbox = list(map(int, bbox))
            # take face crop from face_net detections
            face = cv2_img[max(0, bbox[1] - padding):min(bbox[3] + padding, cv2_img.shape[0] - 1),
                           max(0, bbox[0] - padding):min(bbox[2] + padding, cv2_img.shape[1] - 1)]
            age_preds = self.age_net(face)
            gender_preds = self.gender_net(face)
            gender = self.gender_list[gender_preds.argmax()]
            age = self.age_list[age_preds.argmax()]
            opt_labels.append(
                f"{gender}:{gender_preds.max():.2f}," + f"{age}:{age_preds.max():.2f}")
        return face_dets, opt_labels


def batch_inference_img(opencv_model: OpenCVFaceDetModel, cv2_img: np.ndarray) -> None:
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
