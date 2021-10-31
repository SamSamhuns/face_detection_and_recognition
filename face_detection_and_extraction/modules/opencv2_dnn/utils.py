import cv2
import numpy as np
from modules.common_utils import pad_resize_image, scale_coords


def inference_cv2_model(net, cv2_img, input_size, mean_values):
    resized = pad_resize_image(cv2_img, new_size=input_size)
    # opencv expects BGR format
    blob = cv2.dnn.blobFromImage(resized, 1.0, input_size, mean_values)
    net.setInput(blob)
    detections = net.forward()
    return detections[0][0]


def get_bboxes_confs_areas(detections, det_thres, bbox_area_thres, orig_size, in_size):
    """
    Returns a tuple of bounding boxes, bbox confidence scores and bbox area percentages
    """
    w, h = orig_size
    iw, ih = in_size
    # filter detections below threshold
    detections = detections[detections[:, 2] > det_thres]
    detections[:, 3:7] = detections[:, 3:7] * np.array([iw, ih, iw, ih])
    # only select bboxes with area greater than 0.15% of total area of frame
    total_area = iw * ih
    bbox_area = ((detections[:, 5] - detections[:, 3])
                 * (detections[:, 6] - detections[:, 4]))
    bbox_area_perc = 100 * bbox_area / total_area
    detections = detections[bbox_area_perc > bbox_area_thres]

    bbox_confs = detections[:, 2]
    # rescale detections to orig image size taking the padding into account
    boxes = detections[:, 3:7]
    boxes = scale_coords((ih, iw), boxes, (h, w)).round()

    return boxes, bbox_confs, bbox_area_perc
