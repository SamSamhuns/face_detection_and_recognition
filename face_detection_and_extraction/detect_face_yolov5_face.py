import numpy as np
import torch
import cv2
import os

from modules.common_utils import get_argparse, get_file_type
from modules.common_utils import scale_coords

from modules.yolov5_face.onnx.onnx_utils import preprocess_image, conv_strides_to_anchors, w_non_max_suppression
# from modules.yolov5_face.pytorch.utils.general import non_max_suppression_face


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


def plot_detections(detections, cv2_img, threshold, in_HW, line_thickness=None):
    # plot detections on cv2_img

    # filter weak detections
    detections = detections[detections[..., 4] > threshold]
    boxs = detections[..., :4].numpy()
    confs = detections[..., 4].numpy()

    mh, mw = in_HW
    h, w = cv2_img.shape[:2]
    boxs = scale_coords((mh, mw), boxs, (h, w)).round()
    tl = line_thickness or round(0.002 * (w + h) / 2) + 1
    for i, box in enumerate(boxs):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(cv2_img, (x1, y1), (x2, y2), (0, 0, 255), thickness=max(
            int((w + h) / 600), 1), lineType=cv2.LINE_AA)
        label = f"{confs[i]:.2f}"
        t_size = cv2.getTextSize(
            label, 0, fontScale=tl / 3, thickness=1)[0]
        c2 = x1 + t_size[0] + 3, y1 - t_size[1] - 5
        cv2.rectangle(cv2_img, (x1 - 1, y1), c2,
                      (0, 0, 255), cv2.FILLED, cv2.LINE_AA)
        cv2.putText(cv2_img, label, (x1 + 3, y1 - 4), 0, tl / 3, [255, 255, 255],
                    thickness=1, lineType=cv2.LINE_AA)
    return cv2_img


def inference_pytorch_model(net, cv2_img, input_size, device):
    img = preprocess_image(cv2_img, input_size=input_size)
    img = torch.from_numpy(img).to(device)
    with torch.no_grad():
        pred = net(img)[0]
    pred = non_max_suppression_face(pred, conf_thres=0.3, iou_thres=0.5)
    detections = pred.cpu().numpy() if pred.cuda else pred
    return detections


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

    # pass the image through the network and
    # obtain the detections
    if isinstance(net, torch.nn.Module):  # pytorch inference
        detections = inference_pytorch_model(net, image, input_size)
    else:                                 # onnx inference
        detections = inference_onnx_model(net, image, input_size)

    if detections is not None:
        plot_detections(detections, image, threshold=threshold, in_HW=input_size[::-1])
    # show the output image
    cv2.imshow("output", image)
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

    # When everything done, release the capture
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
                        default=(640, 640),
                        help='Input images are resized to this size (width, height). (default: %(default)s).')
    args = parser.parse_args()
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
