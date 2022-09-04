from typing import Tuple

import cv2

from modules.utils.parser import get_argparse
from modules.utils.files import get_file_type
from modules.opencv2_dnn.model import OpenCVFaceDetAgeGenderModel
from modules.utils.inference import inference_img, inference_vid, inference_webcam


def load_model(
        facedet_model: str = "weights/face_detection_caffe/res10_300x300_ssd_iter_140000.caffemodel",
        facedet_proto: str = "weights/face_detection_caffe/deploy.prototxt.txt",
        age_proto: str = "weights/age_net_caffe/age_deploy.prototxt",
        age_model: str = "weights/age_net_caffe/age_net.caffemodel",
        gender_proto: str = "weights/gender_net_caffe/gender_deploy.prototxt",
        gender_model: str = "weights/gender_net_caffe/gender_net.caffemodel",
        det_thres: float = 0.75,
        bbox_area_thres: float = 0.10,
        model_in_size: Tuple[int, int] = (300, 400),
        device="cpu"
):
    """
    Load face detection, age, and gender estimation models
    """
    if device not in {"cpu", "gpu"}:
        raise NotImplementedError(f"Device {device} is not supported")
    face_net = cv2.dnn.readNet(facedet_model, facedet_proto)
    age_net = cv2.dnn.readNet(age_model, age_proto)
    gender_net = cv2.dnn.readNet(gender_model, gender_proto)
    model_in_size = tuple(map(int, model_in_size))

    if device == "cpu":
        age_net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        gender_net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        face_net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif device == "gpu":
        age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        gender_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")

    return OpenCVFaceDetAgeGenderModel(
        face_net, age_net, gender_net, model_in_size, det_thres, bbox_area_thres)


def main():
    parser = get_argparse(
        description="OpenCV DNN face detection with age and gender estimation")
    parser.add_argument("-p", "--prototxt", dest="prototxt",
                        default="weights/face_detection_caffe/deploy.prototxt.txt",
                        help="Path to 'deploy' prototxt file. (default: %(default)s)")
    parser.add_argument("--is", "--input_size", dest="input_size",
                        nargs=2, default=(300, 400),
                        help='Input images are resized to this (width, height). (default: %(default)s).')
    args = parser.parse_args()
    print("Current Arguments: ", args)
    args.input_size = tuple(map(int, args.input_size))

    # Load networks
    net = load_model(facedet_model=args.model,
                     facedet_proto=args.prototxt,
                     det_thres=args.det_thres,
                     bbox_area_thres=args.bbox_area_thres,
                     model_in_size=args.input_size,
                     device=args.device)
    # choose inference mode
    input_type = get_file_type(args.input_src)
    if input_type == "camera":
        inference_webcam(net, int(args.input_src), "opencv_age_gender")
    elif input_type == "video":
        inference_vid(net, args.input_src, "opencv_age_gender")
    elif input_type == "image":
        inference_img(net, args.input_src, "opencv_age_gender")
    else:
        print("File type or inference mode not recognized. Use --help")


if __name__ == "__main__":
    main()
