import os
import cv2
import numpy as np
from modules.common_utils import get_argparse


class Net(object):
    __slots__ = ["face_net", "age_net", "gender_net",
                 "MODEL_MEAN_VALUES", "age_list", "gender_list"]

    def __init__(self, face_net, age_net, gender_net):
        self.face_net = face_net
        self.age_net = age_net
        self.gender_net = gender_net
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                         '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']


def load_net(facedet_model="weights/opencv_dnn_caffe/res10_300x300_ssd_iter_140000.caffemodel",
             facedet_proto="weights/opencv_dnn_caffe/deploy.prototxt.txt",
             age_proto="weights/age_net/age_deploy.prototxt",
             age_model="weights/age_net/age_net.caffemodel",
             gender_proto="weights/gender_net/gender_deploy.prototxt",
             gender_model="weights/gender_net/gender_net.caffemodel",
             device="cpu"):
    """
    Load face detection, age, and gender estimation models
    """
    if device not in {"cpu", "gpu"}:
        raise NotImplementedError(f"Device {device} is not supported")
    face_net = cv2.dnn.readNet(facedet_model, facedet_proto)
    age_net = cv2.dnn.readNet(age_model, age_proto)
    gender_net = cv2.dnn.readNet(gender_model, gender_proto)

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
        gender_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")

    return Net(face_net, age_net, gender_net)


def inference_and_get_face_boxes(net, cv2_img, conf_threshold=0.7):
    frame_orig = cv2_img.copy()
    fh, fw = frame_orig.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_orig, 1.0, (300, 300), [
                                 104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * fw)
            y1 = int(detections[0, 0, i, 4] * fh)
            x2 = int(detections[0, 0, i, 5] * fw)
            y2 = int(detections[0, 0, i, 6] * fh)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_orig, (x1, y1), (x2, y2),
                          (0, 0, 255), int(round(fh / 150)), 8)
    return frame_orig, bboxes


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

    frame_face, face_bboxes = inference_and_get_face_boxes(
        net.face_net, image, threshold)

    padding = 20
    for bbox in face_bboxes:
        # take face crop from face_net detections
        face = image[max(0, bbox[1] - padding):min(bbox[3] + padding, image.shape[0] - 1),
                     max(0, bbox[0] - padding):min(bbox[2] + padding, image.shape[1] - 1)]
        face_blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), net.MODEL_MEAN_VALUES, swapRB=False)
        # estimate gender
        net.gender_net.setInput(face_blob)
        gender_preds = net.gender_net.forward()
        gender = net.gender_list[gender_preds[0].argmax()]
        # print(f"Gender : {gender}, conf = {gender_preds[0].max():.2f}")

        # estimate age
        net.age_net.setInput(face_blob)
        age_preds = net.age_net.forward()
        age = net.age_list[age_preds[0].argmax()]
        # print(f"Age : {age}, conf = {age_preds[0].max():.2f}")

        gender_age_label = f"{gender}:{gender_preds[0].max():.2f}," + \
                           f"{age}:{age_preds[0].max():.2f}"
        cv2.putText(frame_face, gender_age_label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("output", frame_face)
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

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def inference_webcam(net, threshold):
    inference_vid(net, 0, threshold)


def main():
    parser = get_argparse(
        description="OpenCV DNN face detection with age and gender estimation")
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "gpu"],
                        help="Device to inference on. (default: $(default)s)")
    args = parser.parse_args()

    # Load networks
    net = load_net(facedet_model=args.model,
                   facedet_proto=args.prototxt,
                   device=args.device)
    # choose inference mode
    if args.webcam and args.image is None and args.video is None:
        inference_webcam(net, args.threshold)
    elif args.image and args.video is None and args.webcam is False:
        inference_img(net, args.image, args.threshold)
    elif args.video and args.image is None and args.webcam is False:
        inference_vid(net, args.video, args.threshold)
    else:
        print("Only one mode is allowed")
        print("\tpython detect_face_age_gender.py -w           # webcam mode")
        print("\tpython detect_face_age_gender.py -i img_path  # image mode")
        print("\tpython detect_face_age_gender.py -v vid_path  # video mode")


if __name__ == "__main__":
    main()
