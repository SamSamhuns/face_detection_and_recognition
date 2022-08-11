# common utils should not import from other custom packages
import math
from typing import Tuple

import cv2
import numpy as np


# ##################### image size/coords utils ######################## #


def make_divisible(x, divisor):
    """
    Returns x evenly divisible by divisor
    """
    return math.ceil(x / divisor) * divisor


def check_img_size(img_size: int, s: int = 32):
    """
    Verify img_size is a multiple of stride s
    s = max stride, check with model.stride.max()
    """
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' %
              (img_size, s, new_size))
    return new_size


def pad_resize_image(
        cv2_img: np.ndarray,
        new_size: Tuple[int, int] = (640, 480),
        color: Tuple[int, int, int] = (125, 125, 125)) -> np.ndarray:
    """
    resize and pad image with color if necessary, maintaining orig scale
    args:
        cv2_img: numpy.ndarray = cv2 image
        new_size: tuple(int, int) = (width, height)
        color: tuple(int, int, int) = (B, G, R)
    """
    in_h, in_w = cv2_img.shape[:2]
    new_w, new_h = new_size
    # rescale down
    scale = min(new_w / in_w, new_h / in_h)
    # get new sacled widths and heights
    scale_new_w, scale_new_h = int(in_w * scale), int(in_h * scale)
    resized_img = cv2.resize(cv2_img, (scale_new_w, scale_new_h))
    # calculate deltas for padding
    d_w = max(new_w - scale_new_w, 0)
    d_h = max(new_h - scale_new_h, 0)
    # center image with padding on top/bottom or left/right
    top, bottom = d_h // 2, d_h - (d_h // 2)
    left, right = d_w // 2, d_w - (d_w // 2)
    pad_resized_img = cv2.copyMakeBorder(resized_img,
                                         top, bottom, left, right,
                                         cv2.BORDER_CONSTANT,
                                         value=color)
    return pad_resized_img


def clip_coords(boxes, img_shape):
    """
    Clip bounding xyxy bounding boxes to image shape (height, width)
    """
    if boxes.any():
        if isinstance(boxes, np.ndarray):
            boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
            boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
            boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
            boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2
        else:
            boxes[:, 0].clamp_(0, img_shape[1])  # x1
            boxes[:, 1].clamp_(0, img_shape[0])  # y1
            boxes[:, 2].clamp_(0, img_shape[1])  # x2
            boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape: Tuple[int, int], coords: np.ndarray, img0_shape: Tuple[int, int], ratio_pad=None):
    """
    Rescale coords (xyxy) from img1_shape to img0_shape
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    # x_pad_idxs should be [0, 2] for xyxy
    x_pad_idxs = [i for i in range(coords.shape[-1]) if i & 1 == 0]
    # y_pad_idxs should be [1, 3] for xyxy
    y_pad_idxs = [i for i in range(coords.shape[-1]) if i & 1 == 1]
    coords[:, x_pad_idxs] -= pad[0]  # x padding
    coords[:, y_pad_idxs] -= pad[1]  # y padding
    coords /= gain
    clip_coords(coords, img0_shape)
    return coords


def standardize_image(cv2_img: np.ndarray, new_dtype=np.float32):
    """
    Linearly scales each image in image to have mean 0 and variance 1. or prewhiten image
    """
    if cv2_img.ndim == 4:
        axis = (1, 2, 3)
        size = cv2_img[0].size
    elif cv2_img.ndim == 3:
        axis = (0, 1, 2)
        size = cv2_img.size
    else:
        raise ValueError('Dimension should be 3 or 4')
    mean = np.mean(cv2_img, axis=axis, keepdims=True)
    std = np.std(cv2_img, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    std_img = (cv2_img - mean) / std_adj
    return std_img.astype(new_dtype)


# ##################### bounding box utils ######################### #


def calculate_bbox_iou(bbox1, bbox2):
    """
    bboxes must have coords in fmt (xmin, ymin, xmax, ymax)
    args:
        bbox1: x1min, y1min, x1max, y1max
        bbox2: x2min, y2min, x2max, y2max
    return:
        bounding box IOU between 0 and 1
    """
    x1min, y1min, x1max, y1max = bbox1
    x2min, y2min, x2max, y2max = bbox2
    x_diff = min(x1max, x2max) - max(x1min, x2min)
    y_diff = min(y1max, y2max) - max(y1min, y2min)
    iou = 0
    if x_diff < 0 or y_diff < 0:  # bboxes do not intersect
        return iou
    intersect = x_diff * y_diff
    iou = intersect / (
        ((x1max - x1min) * (y1max - y1min)) + ((x2max - x2min) * (y2max - y2min)) - intersect)
    return iou


def draw_bbox_on_image(cv2_img: np.ndarray, post_dets, line_thickness: int = None, text_bg_alpha: float = 0.5):
    """
    Draw bboxes on cv2 image
        boxes must be 2D list/np array of coords xmin, ymin, xmax, ymax foreach bbox
        confs must be 2D list of confidences foreach corresponding bbox
    """
    boxes = post_dets.boxes
    bbox_confs = post_dets.bbox_confs
    bbox_areas = post_dets.bbox_areas
    bbox_lmarks = post_dets.bbox_lmarks
    bbox_labels = post_dets.bbox_labels
    h, w = cv2_img.shape[:2]
    tl = line_thickness or round(0.002 * (w + h) / 2) + 1

    for i, box in enumerate(boxes):
        if bbox_areas is None:
            label = f"{bbox_confs[i]:.2f}"
        else:
            label = f"{bbox_confs[i]:.2f}_{bbox_areas[i]:.2f}"
        xmin, ymin, xmax, ymax = map(int, box)
        xmin, ymin, xmax, ymax = (
            max(xmin, 0), max(ymin, 0), min(xmax, w), min(ymax, h))
        # draw bbox on image
        cv2.rectangle(cv2_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=max(
            int((w + h) / 600), 1), lineType=cv2.LINE_AA)

        # draw landmarks if provided
        if bbox_lmarks is not None and bbox_lmarks.any():
            for li in range(0, len(bbox_lmarks[i]), 2):
                cx, cy = int(bbox_lmarks[i][li]), int(bbox_lmarks[i][li + 1])
                cv2.circle(cv2_img, (cx, cy), radius=3, color=(0, 0, 255), thickness=1)

        # Add optional label to bbox if provided
        if bbox_labels:
            label += str(bbox_labels[i])

        # draw rect covering text
        t_size = cv2.getTextSize(
            label, 0, fontScale=tl / 3, thickness=1)[0]
        c2 = xmin + t_size[0] + 3, ymin - t_size[1] - 5
        color = (0, 0, 0)
        if text_bg_alpha == 0.0:
            cv2.rectangle(cv2_img, (xmin - 1, ymin), c2,
                          color, cv2.FILLED, cv2.LINE_AA)
        else:
            # Transparent text background
            alphaReserve = text_bg_alpha  # 0: opaque 1: transparent
            BChannel, GChannel, RChannel = color
            xMin, yMin = int(xmin - 1), int(ymin - t_size[1] - 3)
            xMax, yMax = int(xmin + t_size[0]), int(ymin)
            cv2_img[yMin:yMax, xMin:xMax, 0] = cv2_img[yMin:yMax,
                                                       xMin:xMax, 0] * alphaReserve + BChannel * (1 - alphaReserve)
            cv2_img[yMin:yMax, xMin:xMax, 1] = cv2_img[yMin:yMax,
                                                       xMin:xMax, 1] * alphaReserve + GChannel * (1 - alphaReserve)
            cv2_img[yMin:yMax, xMin:xMax, 2] = cv2_img[yMin:yMax,
                                                       xMin:xMax, 2] * alphaReserve + RChannel * (1 - alphaReserve)
        # draw label text
        cv2.putText(cv2_img, label, (xmin + 3, ymin - 4), 0, fontScale=tl / 4,
                    color=[255, 255, 255], thickness=1, lineType=cv2.LINE_AA)


def get_distinct_rgb_color(index: int):
    """
    Get a RGB color from a pre-defined colors list
    """
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255), (0, 0, 0), (128, 0, 0),
                  (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128),
                  (0, 128, 128), (128, 128, 128), (192, 0, 0), (0, 192, 0),
                  (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192),
                  (192, 192, 192), (64, 0, 0), (0, 64, 0), (0, 0, 64),
                  (64, 64, 0), (64, 0, 64), (0, 64, 64), (64, 64, 64),
                  (32, 0, 0), (0, 32, 0), (0, 0, 32), (32, 32, 0),
                  (32, 0, 32), (0, 32, 32), (32, 32, 32), (96, 0, 0),
                  (0, 96, 0), (0, 0, 96), (96, 96, 0), (96, 0, 96), (0, 96, 96),
                  (96, 96, 96), (160, 0, 0), (0, 160, 0), (0, 0, 160),
                  (160, 160, 0), (160, 0, 160), (0, 160, 160), (160, 160, 160),
                  (224, 0, 0), (0, 224, 0), (0, 0, 224), (224, 224, 0),
                  (224, 0, 224), (0, 224, 224), (224, 224, 224)]
    if index >= len(color_list):
        print(
            f"WARNING:color index {index} exceeds available number of colors {len(color_list)}. Cycling colors now")
        index %= len(color_list)

    return color_list[index]
