import numpy as np
import cv2
import os

from modules.utils.files import get_file_type
from modules.utils.parser import get_argparse
from modules.opencv2_dnn.model import OpenCVModel
from modules.models.inference import inference_img, inference_vid, inference_webcam


def load_model(model, prototxt, det_thres, bbox_area_thres, input_size=(300, 300), device="cpu"):
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
    return OpenCVModel(face_net, input_size, det_thres, bbox_area_thres)


def batch_inference_img(net, cv2_img):
    """Reference func for batched OpenCV DNN inference"""
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
    parser.add_argument("--is", "--input_size", dest="input_size",
                        default=(300, 400),
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
