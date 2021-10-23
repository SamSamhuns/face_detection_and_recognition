import numpy as np
import torch
import cv2
import os

from modules.common_utils import get_argparse, get_file_type
from modules.common_utils import scale_coords, draw_bbox_on_image

from modules.yolov5_face.onnx.onnx_utils import check_img_size, preprocess_image, conv_strides_to_anchors, w_non_max_suppression


def load_net(model):
    # load face detection model
    fpath, fext = os.path.splitext(model)
    if fext == ".pth":
        from modules.yolov5_face.pytorch import attempt_load

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = attempt_load(model, device)
    elif fext == ".onnx":
        import onnxruntime
        net = onnxruntime.InferenceSession(model)
    else:
        raise NotImplementedError(
            f"[ERROR] model with extension {fext} not implemented")
    return net


def plot_detections(detections, cv2_img, threshold, in_size_WH):
    # filter detections below threshold
    detections = detections[detections[..., 4] > threshold]
    boxes = detections[..., :4].numpy()
    confs = detections[..., 4].numpy()

    mw, mh = in_size_WH
    h, w = cv2_img.shape[:2]
    # rescale detections to orig image size taking the padding into account
    boxes = scale_coords((mh, mw), boxes, (h, w)).round()
    draw_bbox_on_image(cv2_img, boxes, confs)
    return cv2_img


def inference_onnx_model(net, cv2_img, input_size):
    img = preprocess_image(cv2_img, input_size=input_size)
    outputs = net.run(None, {"images": img})
    outputx = conv_strides_to_anchors(outputs, "cpu")
    detections = w_non_max_suppression(
        outputx, num_classes=1, conf_thres=0.4, nms_thres=0.3)
    return detections[0]


def inference_img(net, img, input_size, threshold, waitKey_val=0):
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
    detections = inference_onnx_model(net, image, input_size)
    if detections is not None:
        plot_detections(detections, image, threshold=threshold, in_size_WH=input_size)

    cv2.imshow("YOLOv5 face", image)
    cv2.waitKey(waitKey_val)


def inference_vid(net, vid, input_size, threshold):
    cap = cv2.VideoCapture(vid)
    ret, frame = cap.read()

    while ret:
        # inference and display the resulting frame
        inference_img(net, frame, input_size, threshold, waitKey_val=5)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()


def inference_webcam(net, cam_index, input_size, threshold):
    inference_vid(net, cam_index, input_size, threshold)


def main():
    parser = get_argparse(
        description="Blazeface face detection", conflict_handler='resolve')
    parser.remove_argument("prototxt")
    parser.add_argument("-m", "--model",
                        default="weights/yolov5s/yolov5s-face.onnx",
                        help='Path to weight file (.pth/.onnx). (default: %(default)s).')
    parser.add_argument("-is", "--input_size",
                        nargs=2,
                        default=(640, 640),
                        help='Input images are resized to this size (width, height). (default: %(default)s).')
    args = parser.parse_args()
    args.input_size = tuple(map(int, args.input_size))
    # ensure input size is of correct mult
    args.input_size = tuple(map(check_img_size, args.input_size))
    print("Current Arguments: ", args)

    net = load_net(args.model)
    # choose inference mode
    input_type = get_file_type(args.input_src)
    if input_type == "camera":
        inference_webcam(net, int(args.input_src), args.input_size, args.threshold)
    elif input_type == "video":
        inference_vid(net, args.input_src, args.input_size, args.threshold)
    elif input_type == "image":
        inference_img(net, args.input_src, args.input_size, args.threshold)
    else:
        print("File type or inference mode not recognized. Use --help")


if __name__ == "__main__":
    main()
