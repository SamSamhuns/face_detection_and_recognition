import cv2
import os

from modules.utils.files import get_file_type
from modules.utils.parser import get_argparse
from modules.opencv2_dnn.model import OpenCVModel
from modules.models.inference import inference_img, inference_vid, inference_webcam


def load_model(model_path: str, prototxt_path, det_thres: float, bbox_area_thres: float, input_size: str, device: str = "cpu"):
    # load face detection model
    if device not in {"cpu", "gpu"}:
        raise NotImplementedError(f"Device {device} is not supported")

    fname, fext = os.path.splitext(model_path)
    if fext == ".caffemodel":
        face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    elif fext == ".pb":
        face_net = cv2.dnn.readNetFromTensorflow(model_path, prototxt_path)
    else:
        raise NotImplementedError(
            f"[ERROR] model with extension {fext} not implemented")

    if device == "cpu":
        face_net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif device == "gpu":
        face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")
    input_size = tuple(map(int, input_size.split(',')))  # conv "W_str, H_str" to (W_str, H_str)
    return OpenCVModel(face_net, input_size, det_thres, bbox_area_thres)


def main():
    parser = get_argparse(description="OpenCV DNN face detection")
    parser.add_argument("--is", "--input_size", dest="input_size",
                        default="300,400",
                        help='Input images are resized to this (width, height). (default: %(default)s).')
    parser.add_argument("-p", "--prototxt", dest="prototxt",
                        default="weights/face_detection_caffe/deploy.prototxt.txt",
                        help="Path to 'deploy' prototxt file. (default: %(default)s)")
    args = parser.parse_args()
    print("Current Arguments: ", args)

    net = load_model(args.model,
                     args.prototxt,
                     det_thres=args.det_thres,
                     bbox_area_thres=args.bbox_area_thres,
                     input_size=args.input_size,
                     device=args.device)
    # choose inference mode
    input_type = get_file_type(args.input_src)
    if input_type == "camera":
        inference_webcam(net, int(args.input_src), "opencv_dnn")
    elif input_type == "video":
        inference_vid(net, args.input_src, "opencv_dnn")
    elif input_type == "image":
        inference_img(net, args.input_src, "opencv_dnn")
    else:
        print("File type or inference mode not recognized. Use --help")


if __name__ == "__main__":
    main()
