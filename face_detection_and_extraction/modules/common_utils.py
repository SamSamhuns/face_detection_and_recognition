import argparse
from typing import List

import cv2
import mimetypes
import numpy as np


class ArgumentParserMod(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def remove_argument(self, arg: str):
        """
        Remove argument from argparse object
        args:
            arg: argument name without leading dashes
        """
        for action in self._actions:
            if (vars(action)['option_strings']
                and vars(action)['option_strings'][0] == arg) \
                    or vars(action)['dest'] == arg:
                self._remove_action(action)

        for action in self._action_groups:
            vars_action = vars(action)
            var_group_actions = vars_action['_group_actions']
            for x in var_group_actions:
                if x.dest == arg:
                    var_group_actions.remove(x)
                    return

    def remove_arguments(self, arg_list: List[str]):
        """
        Remove list of arguments from argparse object
        args:
        """
        [self.remove_argument(arg) for arg in arg_list]


def get_argparse(*args, **kwargs):
    """
    get base argparse arguments
        remove arguments with parser.remove_argparse_option(...)
        add new arguments with parser.add_argument(...)
    """
    parser = ArgumentParserMod(*args, **kwargs)
    parser.add_argument("-i", "--input_src", default='0',
                        help="Path to input image/video/cam_index:\n" +
                             "\t IMAGE_DDOE       -i <PATH_TO_IMG>\n" +
                             "\t VIDEO_MODE       -i <PATH_TO_VID>\n" +
                             "\t CAM MODE:Default -i <CAM_INDEX>  -i 0 (for webcam)\n")
    parser.add_argument("-m", "--model",
                        default="weights/face-detection-caffe/res10_300x300_ssd_iter_140000.caffemodel",
                        help='Path to model file. (default: %(default)s)')
    parser.add_argument("-p", "--prototxt",
                        default="weights/face-detection-caffe/deploy.prototxt.txt",
                        help="Path to 'deploy' prototxt file. (default: %(default)s)")
    parser.add_argument("-dt", "--det_thres",
                        type=float, default=0.75,
                        help='score to filter weak detections. (default: %(default)s)')
    parser.add_argument("-at", "--bbox_area_thres",
                        type=float, default=0.15,
                        help='score to filter bboxes that cover small area %. (default: %(default)s)')

    return parser


def fix_path_for_globbing(dir):
    """ Add * at the end of paths for proper globbing
    """
    if dir[-1] == '/':         # data/
        dir += '*'
    elif dir[-1] != '*':       # data
        dir += '/*'
    else:                      # data/*
        dir = dir
    return dir


def get_file_type(file_src):
    """
    Returns if a file is image/video/camera/None based on extension or int type
    """
    if file_src.isnumeric():
        return 'camera'
    mimetypes.init()
    mimestart = mimetypes.guess_type(file_src)[0]

    file_type = None
    if mimestart is not None:
        mimestart = mimestart.split('/')[0]
        if mimestart in ['video', 'image']:
            file_type = mimestart
    return file_type


def get_distinct_rgb_color(index):
    """
    Get a RGB color from a pre-defined colors list
    """
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                  (255, 0, 255),  (0, 255, 255), (0, 0, 0), (128, 0, 0),
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


def pad_resize_image(cv2_img, new_size=(640, 480), color=(125, 125, 125)) -> np.ndarray:
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
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, np.ndarray):
        boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
        boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
        boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
        boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2
    else:  # torch.Tensor
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


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
    iou = intersect / (((x1max - x1min) * (y1max - y1min)) +
                       ((x2max - x2min) * (y2max - y2min)) - intersect)
    return iou


def draw_bbox_on_image(cv2_img, boxes, confs):
    """
    Draw bboxes on cv2 image
        boxes must be 2D list/np array of coords xmin, ymin, xmax, ymax foreach bbox
        confs must be 2D list of confidences foreach corresponding bbox
    """
    h, w = cv2_img.shape[:2]
    tl = round(0.002 * (w + h) / 2) + 1

    for i, box in enumerate(boxes):
        label = f"{confs[i]:.2f}"
        # draw bbox on image
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(cv2_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=max(
            int((w + h) / 600), 1), lineType=cv2.LINE_AA)
        t_size = cv2.getTextSize(
            label, 0, fontScale=tl / 3, thickness=1)[0]
        # draw label text along with text bg rect
        c2 = xmin + t_size[0] + 3, ymin - t_size[1] - 5
        cv2.rectangle(cv2_img, (xmin - 1, ymin), c2,
                      (0, 0, 255), cv2.FILLED, cv2.LINE_AA)
        cv2.putText(cv2_img, label, (xmin + 3, ymin - 4), 0, tl / 3, [255, 255, 255],
                    thickness=1, lineType=cv2.LINE_AA)
