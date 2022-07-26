from typing import Tuple, List, Any, Optional

import os
import cv2
import numpy as np

from modules.models.base import Model, PostProcessedDetection
from modules.utils.image import scale_coords, draw_bbox_on_image


def get_dets_bboxes_confs_lmarks_areas(
        dets: np.ndarray,
        orig_size: Tuple[int, int],
        in_size: Tuple[int, int],
        det_thres: float,
        bbox_area_thres: float,
        opt_labels: Optional[List[Any]] = None,
) -> PostProcessedDetection:
    """
    Returns a PostProcessedDetections object containing bbox, bbox confidence scores, bbox areas and optionally face landmarks
    Arguments:
        dets: np.ndarray[np.ndarray[xmin, ymin, xmax, ymax, conf[, lmarks...]], ...] = bounding boxes must be normalized
        det_thres: float = detection threshold
        bbox_area_thres: float = bounding box area threshold as compared to orig image size
        orig_size: Tuple[int, int] = original image size (width, height)
        in_size: Tuple[int, int] = model input image size (width, height)
        opt_labels: List[Any] list of optional labels to add to bbox
    """
    post_dets_args = {}
    w, h = orig_size
    iw, ih = in_size

    # filter dets below threshold
    dets = dets[dets[:, 4] > det_thres]
    # denorm bounding boxes to model input_size
    dets[:, :4] = dets[:, :4] * np.array([iw, ih, iw, ih])
    # only select bboxes with area greater than bbox_area_thres of total area of frame
    total_area = iw * ih
    bbox_area = ((dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1]))
    bbox_area_perc = bbox_area / total_area
    bbox_area_perc_filter = 100 * bbox_area_perc > bbox_area_thres
    dets = dets[bbox_area_perc_filter]
    # select bbox_area_percs higher than bbox_area_thres
    bbox_area_perc = bbox_area_perc[bbox_area_perc_filter]
    post_dets_args["bbox_areas"] = bbox_area_perc

    bbox_confs = dets[:, 4]
    post_dets_args["bbox_confs"] = bbox_confs
    # rescale dets to orig image size taking the padding into account
    boxes = dets[:, :4]
    boxes = scale_coords((ih, iw), boxes, (h, w)).round()
    post_dets_args["boxes"] = boxes

    # add face-landmark coords if dets provides those values
    bbox_lmarks = dets[:, 5:]
    post_dets_args["bbox_lmarks"] = scale_coords((ih, iw), bbox_lmarks, (h, w)).round()
    # add optional labels
    post_dets_args["bbox_labels"] = opt_labels

    return PostProcessedDetection(**post_dets_args)


def inference_img(
        net: Model,
        img: np.ndarray,
        wname: str = "Output",
        waitKey_val: int = 0) -> None:
    """Run inference on an image and display it"""
    if isinstance(img, str):
        if os.path.exists(img):
            image = cv2.imread(img)
        else:
            raise FileNotFoundError(f"{img} does not exist")
    elif isinstance(img, np.ndarray):
        image = img
    else:
        raise Exception("image cannot be read")
    # dets will always be a 2D numpy array [[xmin, ymin, xmax, ymax, conf[, lmarks]], ..]
    # opt_labels are optional labels for bounding boxes such as age, gender info, etc
    opt_labels = []
    if net.returns_opt_labels:
        dets, opt_labels = net(image)
    else:
        dets = net(image)

    h, w = image.shape[:2]
    iw, ih = net.input_size

    # get bounding boxes, conf scores, face-landmarks if any and face areas
    post_dets = get_dets_bboxes_confs_lmarks_areas(
        dets, (w, h), (iw, ih), net.det_thres, net.bbox_area_thres, opt_labels)
    draw_bbox_on_image(image, post_dets)

    cv2.imshow(wname, image)
    cv2.waitKey(waitKey_val)


def inference_vid(
        net: Model,
        vid: str,
        wname: str = "Output") -> None:
    """Run inference on video and display it"""
    cap = cv2.VideoCapture(vid)
    ret, frame = cap.read()

    while ret:
        # inference and display the resulting frame
        inference_img(net, frame, wname=wname, waitKey_val=5)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()


def inference_webcam(
        net: Model,
        cam_index: int,
        wname: str = "Output") -> None:
    """Run inference on webcam video input and display it"""
    inference_vid(net, vid=cam_index, wname=wname)
