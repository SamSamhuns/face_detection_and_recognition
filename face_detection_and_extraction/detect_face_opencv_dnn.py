import numpy as np
import cv2
import os

from modules.common_utils import get_argparse, get_file_type, draw_bbox_on_image
from modules.opencv2_dnn.utils import inference_cv2_model, get_bboxes_confs_areas


class Net(object):
    __slots__ = ["face_net", "det_thres", "bbox_area_thres",
                 "FACE_MODEL_MEAN_VALUES", "FACE_MODEL_INPUT_SIZE"]

    def __init__(self, face_net, det_thres, bbox_area_thres, model_in_size):
        self.face_net = face_net
        self.det_thres = det_thres
        self.bbox_area_thres = bbox_area_thres
        self.FACE_MODEL_INPUT_SIZE = model_in_size  # (width, height)
        self.FACE_MODEL_MEAN_VALUES = (104.0, 117.0, 123.0)


def load_net(model, prototxt, det_thres, bbox_area_thres, model_in_size=(300, 300), device="cpu"):
    # load face detection model
    if device not in {"cpu", "gpu"}:
        raise NotImplementedError(f"Device {device} is not supported")

    fname, fext = os.path.splitext(model)
    if fext == ".caffemodel":
        face_net = cv2.dnn.readNetFromCaffe(prototxt, model)
    elif fext == ".pb":
        face_net = cv2.dnn.readNetFromTensorflow(model, prototxt)
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
    return Net(face_net, det_thres, bbox_area_thres, model_in_size)


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

    h, w = image.shape[:2]
    iw, ih = net.FACE_MODEL_INPUT_SIZE

    # pass the blob through the network and
    # obtain the detections and predictions
    detections = inference_cv2_model(net.face_net,
                                     image,
                                     net.FACE_MODEL_INPUT_SIZE,
                                     net.FACE_MODEL_MEAN_VALUES)
    boxes, confs, areas = get_bboxes_confs_areas(
        detections, net.det_thres, net.bbox_area_thres, (w, h), (iw, ih))
    draw_bbox_on_image(image, boxes, confs, areas)

    cv2.imshow("opencv_dnn", image)
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


def batch_inference_img(net, cv2_img):
    """reference func for batched DNN inference
    """
    image = cv2_img
    blob = cv2.dnn.blobFromImages(
        image, 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.face_net.setInput(blob)
    detections = net.face_net.forward()

    count = 0
    threshold = 0.5
    img_idx = 0
    (h, w) = image[0].shape[:2]

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability)
        # associated with the prediction
        confidence = detections[0, 0, i, 2]
        det_img_idx = int(detections[0, 0, i, 0])

        # filter weak detections
        if confidence > threshold:
            count += 1
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image[det_img_idx], (startX, startY),
                          (endX, endY), (0, 0, 255), 2)
            cv2.putText(image[det_img_idx], text, (startX, y),
                        cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 0, 255), 2)
        if (i + 1) % 200 == 0:
            cv2.imshow("output", image[img_idx])
            cv2.waitKey(0)
            img_idx += 1


def main():
    parser = get_argparse(description="OpenCV DNN face detection")
    parser.add_argument("-is", "--input_size",
                        default=(300, 400),
                        help='Input images are resized to this (width, height). (default: %(default)s).')
    args = parser.parse_args()
    print("Current Arguments: ", args)

    net = load_net(args.model,
                   args.prototxt,
                   det_thres=args.det_thres,
                   bbox_area_thres=args.bbox_area_thres,
                   model_in_size=args.input_size,
                   device=args.device)
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
