import torch
import numpy as np

from utils.general import non_max_suppression_face
from models.experimental import attempt_load
from modules.utils.image import pad_resize_image, check_img_size


def preprocess_image(cv2_image, input_size=(640, 640)):
    """preprocesses a cv2_image BGR
    args:
        cv2_image = cv2 image
        in_size: in_width, in_height
    """
    cv2_image = cv2_image[..., ::-1]  # BGR2RGB
    # make sure img dims are divisible by model stride
    in_w, in_h = tuple(map(check_img_size, input_size))
    pad_resized = pad_resize_image(cv2_image, (in_w, in_h))
    img = np.transpose(pad_resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    return img


def inference_pytorch_model_yolov5_face(net, cv2_img, input_size):
    # note any other kwargs are ignored
    resized = preprocess_image(cv2_img, input_size=input_size)
    outputs = net(torch.from_numpy(resized))[0]
    detections = non_max_suppression_face(
        outputs, conf_thres=0.4, iou_thres=0.5)
    return detections[0]
