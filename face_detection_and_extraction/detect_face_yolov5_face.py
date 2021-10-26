import numpy as np
import torch
import cv2
import os

from modules.common_utils import get_argparse, get_file_type
from modules.common_utils import check_img_size, draw_bbox_on_image
from modules.yolov5_face.onnx.onnx_utils import inference_onnx_model_yolov5_face, get_bboxes_and_confs


class Net(object):
    __slots__ = ["face_net", "det_thres", "bbox_area_thres", "model_in_size"]

    def __init__(self, face_net, det_thres, bbox_area_thres, model_in_size):
        self.face_net = face_net
        self.det_thres = det_thres
        self.bbox_area_thres = bbox_area_thres
        # in_size = (width, height), conv to int
        model_in_size = tuple(map(int, model_in_size))
        # input size must be multiple of max stride 32 for yolov5 models
        self.model_in_size = tuple(map(check_img_size, model_in_size))


def load_net(model, det_thres, bbox_area_thres, model_in_size):
    # load face detection model
    fpath, fext = os.path.splitext(model)
    if fext == ".pth":
        raise NotImplementedError("pytorch model inference not setup yet")
        from modules.yolov5_face.pytorch import attempt_load

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = attempt_load(model, device)
    elif fext == ".onnx":
        import onnxruntime
        net = onnxruntime.InferenceSession(model)
    else:
        raise NotImplementedError(
            f"[ERROR] model with extension {fext} not implemented")
    return Net(net, det_thres, bbox_area_thres, model_in_size)


def inference_img(net, img, waitKey_val=0):
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    if isinstance(img, str):
        if os.path.exists(img):
            image = cv2.imread(img)
        else:
            raise FileNotFoundError(f"{img} does not exist")
    elif isinstance(img, np.ndarray):
        image = img
    else:
        raise Exception("image cannot be read")

    # pass the image through the network and get detections
    detections = inference_onnx_model_yolov5_face(net.face_net, image, net.model_in_size)
    if detections is not None:
        iw, ih = net.model_in_size
        h, w = image.shape[:2]
        boxes, confs = get_bboxes_and_confs(
            detections, net.det_thres, net.bbox_area_thres, orig_size=(w, h), in_size=(iw, ih))
        draw_bbox_on_image(image, boxes, confs)

    cv2.imshow("YOLOv5 face", image)
    cv2.waitKey(waitKey_val)


def inference_vid(net, vid):
    cap = cv2.VideoCapture(vid)
    ret, frame = cap.read()

    while ret:
        # inference and display the resulting frame
        inference_img(net, frame, waitKey_val=5)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()


def inference_webcam(net, cam_index):
    inference_vid(net, cam_index)


def main():
    parser = get_argparse(
        description="YOLOv5-face face detection", conflict_handler='resolve')
    parser.remove_argument("prototxt")
    parser.add_argument("-m", "--model",
                        default="weights/yolov5s/yolov5s-face.onnx",
                        help='Path to weight file (.pth/.onnx). (default: %(default)s).')
    parser.add_argument("-is", "--input_size",
                        nargs=2,
                        default=(640, 640),
                        help='Input images are resized to this size (width, height). (default: %(default)s).')
    args = parser.parse_args()
    print("Current Arguments: ", args)

    net = load_net(args.model, args.det_thres, args.bbox_area_thres, args.input_size)
    # choose inference mode
    input_type = get_file_type(args.input_src)
    if input_type == "camera":
        inference_webcam(net, int(args.input_src))
    elif input_type == "video":
        inference_vid(net, args.input_src)
    elif input_type == "image":
        inference_img(net, args.input_src)
    else:
        print("File type or inference mode not recognized. Use --help")


if __name__ == "__main__":
    main()
