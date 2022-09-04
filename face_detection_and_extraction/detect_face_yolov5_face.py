import os
import sys
from typing import Tuple

import torch

from modules.utils.parser import get_argparse
from modules.utils.files import get_file_type
from modules.yolov5_face.model import YOLOV5FaceModel
from modules.utils.inference import inference_img, inference_vid, inference_webcam


def load_model(
    model_path: str,
    det_thres: float,
    bbox_area_thres: float,
    model_in_size: Tuple[int, int],
    device: str
):
    # load face detection model
    _, fext = os.path.splitext(model_path)
    if fext in {".pt", ".pth"}:
        sys.path.append("modules/yolov5_face/pytorch")
        from modules.yolov5_face.pytorch import attempt_load, inference_pytorch_model_yolov5_face as inf_func

        device = torch.device(
            device if "cuda" in device and torch.cuda.is_available() else "cpu")
        net = attempt_load(model_path, device)
    elif fext == ".onnx":
        import onnxruntime
        from modules.yolov5_face.onnx.onnx_utils import inference_onnx_model_yolov5_face as inf_func

        net = onnxruntime.InferenceSession(model_path)
    else:
        raise NotImplementedError(
            f"[ERROR] model with extension {fext} not implemented")

    return YOLOV5FaceModel(net, det_thres, bbox_area_thres, inf_func, model_in_size)


def main():
    parser = get_argparse(
        description="YOLOv5-face face detection", conflict_handler='resolve')
    parser.remove_argument(["model"])
    parser.add_argument("--md", "--model", dest="model",
                        default="weights/yolov5s/yolov5s-face.onnx",
                        help='Path to weight file (.pth/.onnx). (default: %(default)s).')
    parser.add_argument("--is", "--input_size", dest="input_size",
                        nargs=2, default=(640, 640),
                        help='Input images are resized to this size (width, height). (default: %(default)s).')
    args = parser.parse_args()
    print("Current Arguments: ", args)
    args.input_size = tuple(map(int, args.input_size))

    net = load_model(args.model, args.det_thres, args.bbox_area_thres,
                     args.input_size, args.device)
    # choose inference mode
    input_type = get_file_type(args.input_src)
    if input_type == "camera":
        inference_webcam(net, int(args.input_src), "yolov5_face")
    elif input_type == "video":
        inference_vid(net, args.input_src, "yolov5_face")
    elif input_type == "image":
        inference_img(net, args.input_src, "yolov5_face")
    else:
        print("File type or inference mode not recognized. Use --help")


if __name__ == "__main__":
    main()
