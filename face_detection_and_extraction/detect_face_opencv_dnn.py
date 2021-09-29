import numpy as np
import cv2
import os

from modules.common_utils import get_argparse, get_file_type
from modules.common_utils import pad_resize_image, scale_coords


class Net(object):
    __slots__ = ["face_net", "FACE_MODEL_MEAN_VALUES", "FACE_MODEL_INPUT_SIZE"]

    def __init__(self, face_net, model_in_size):
        self.face_net = face_net
        self.FACE_MODEL_INPUT_SIZE = model_in_size  # (width, height)
        self.FACE_MODEL_MEAN_VALUES = (104.0, 117.0, 123.0)


def load_net(model, prototxt, model_in_size=(300, 300), device="cpu"):
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
    return Net(face_net, model_in_size)


def inference_model(net, cv2_img, new_size):
    resized = pad_resize_image(cv2_img, new_size)
    # opencv expects BGR format
    blob = cv2.dnn.blobFromImage(resized, 1.0, new_size, (104.0, 117.0, 123.0))
    net.face_net.setInput(blob)
    faces = net.face_net.forward()
    return faces


def inference_img(net, img, threshold, waitKey_val=0):
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
    nw, nh = net.FACE_MODEL_INPUT_SIZE

    # pass the blob through the network and
    # obtain the detections and predictions
    detections = inference_model(net, image, net.FACE_MODEL_INPUT_SIZE)[0][0]

    # filter dtections below threshold
    detections = detections[detections[:, 2] > threshold]
    # rescale detections to orig image size taking the padding into account
    boxes = detections[:, 3:7] * np.array([nw, nh, nw, nh])
    boxes = scale_coords((nh, nw), boxes, (h, w)).round()

    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box.astype('int')
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 125, 255), 3)
        text = f'label:{int(detections[i, 1])}, conf:{detections[i, 2]:.2f}'
        cv2.putText(image, text, (xmin, ymin - 7),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 125, 255), 1)
    # print(f"Num of faces detected = {i} ")
    cv2.imshow("opencv_dnn", image)
    cv2.waitKey(waitKey_val)


def inference_vid(net, vid, threshold):
    cap = cv2.VideoCapture(vid)
    ret, frame = cap.read()

    while ret:
        # inference and display the resulting frame
        inference_img(net, frame, threshold, waitKey_val=5)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()


def inference_webcam(net, cam_index, threshold):
    inference_vid(net, cam_index, threshold)


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
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "gpu"],
                        help="Device to inference on. (default: $(default)s)")
    parser.add_argument("-is", "--input_size",
                        default=(300, 400),
                        help='Input images are resized to this (width, height). (default: %(default)s).')
    args = parser.parse_args()
    print("Current Arguments: ", args)

    net = load_net(args.model,
                   args.prototxt,
                   model_in_size=args.input_size,
                   device=args.device)
    # choose inference mode
    input_type = get_file_type(args.input_src)
    if input_type == "camera":
        inference_webcam(net, int(args.input_src), args.threshold)
    elif input_type == "video":
        inference_vid(net, args.input_src, args.threshold)
    elif input_type == "image":
        inference_img(net, args.input_src, args.threshold)
    else:
        print("File type or inference mode not recognized. Use --help")


if __name__ == "__main__":
    main()
