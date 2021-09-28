import time
import argparse

import cv2
import torch
import numpy as np
import onnxruntime

from onnx_utils import check_img_size, preprocess_image
from onnx_utils import disp_output, conv_strides_to_anchors
from onnx_utils import non_max_suppression, w_non_max_suppression


@torch.no_grad()
def detect_onnx(img_path,
                onnx_path,
                threshold,
                num_classes=1,
                official=False):
    session = onnxruntime.InferenceSession(onnx_path)
    model_batch_size = session.get_inputs()[0].shape[0]
    model_h = session.get_inputs()[0].shape[2]
    model_w = session.get_inputs()[0].shape[3]
    in_w = 640 if (model_w is None or isinstance(model_w, str)) else model_w
    in_h = 640 if (model_h is None or isinstance(model_h, str)) else model_h
    # make sure image size is divisible by strides
    in_w = check_img_size(in_w)
    in_h = check_img_size(in_h)
    print("Input Layer: ", session.get_inputs()[0].name)
    print("Output Layer: ", session.get_outputs()[0].name)
    print("Model Input Shape: ", session.get_inputs()[0].shape)
    print("Model Output Shape: ", session.get_outputs()[0].shape)

    start_time = time.time()
    cv2_img = cv2.imread(img_path)
    model_input = preprocess_image(cv2_img, input_size=(in_w, in_h))
    batch_size = model_input.shape[0] if isinstance(
        model_batch_size, str) else model_batch_size
    input_name = session.get_inputs()[0].name

    # inference
    start = time.time()
    outputs = session.run(None, {input_name: model_input})
    end = time.time()

    inf_time = end - start
    print('Inference Time: {} Seconds Single Image'.format(inf_time))
    fps = 1. / (end - start)
    print('Estimated Inference FPS: {} FPS Single Image'.format(fps))

    batch_detections = []
    # model.model[-1].export = boolean ---> True:3 False:4
    if official and len(outputs) == 4:  # recommended
        # model.model[-1].export = False ---> outputs[0] (1, xxxx, 85)
        # Use the official code directly
        batch_detections = torch.from_numpy(np.array(outputs[0]))
        batch_detections = non_max_suppression(
            batch_detections, conf_thres=0.4, iou_thres=0.5, agnostic=False)
    else:
        outputx = conv_strides_to_anchors(outputs, "cpu")
        batch_detections = w_non_max_suppression(
            outputx, num_classes, conf_thres=0.4, nms_thres=0.3)

    elapse_time = time.time() - start_time
    print(f'Total Elapsed Time: {elapse_time:.3f} Seconds'.format())
    print(f'Final Estimated FPS: {1 / (elapse_time):.2f}')
    disp_output(batch_detections[0], cv2_img,
                threshold=threshold, model_in_HW=(in_h, in_w),
                line_thickness=None, text_bg_alpha=0.0)


def parse_arguments(desc):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--img_path',
                        required=True,  type=str,
                        help='Path to input image file')
    parser.add_argument('-ox', '--onnx_path',
                        default="weights/yolov5n/yolov5n-face.onnx",  type=str,
                        help='Path to ONNX model. (default: %(default)s)')
    parser.add_argument('-t', '--threshold',
                        default=0.6,  type=float,
                        help='Detection Threshold. (default: %(default)s)')
    parser.add_argument('-c', '--num_classes',
                        default=1,  type=int,
                        help='Number of classes. (default: %(default)s)')
    parser.add_argument('-o', '--official',
                        action="store_true",
                        help='Flag to use official yolov5 post-processing')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments("YoloV5 onnx demo")
    t1 = time.time()
    detect_onnx(**vars(args))
