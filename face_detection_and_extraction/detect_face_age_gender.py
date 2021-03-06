import os
import cv2
import numpy as np

from modules.utils.parser import get_argparse
from modules.utils.files import get_file_type
from modules.models.base import PostProcessedDetection
from modules.utils.image import draw_bbox_on_image, pad_resize_image, scale_coords


class Net(object):
    __slots__ = ["face_net", "age_net", "gender_net", "det_thres", "bbox_area_thres",
                 "FACE_MODEL_MEAN_VALUES", "FACE_MODEL_INPUT_SIZE",
                 "AGE_GENDER_INPUT_SIZE", "AGE_GENDER_MODEL_MEAN_VALUES",
                 "age_list", "gender_list"]

    def __init__(self, face_net, age_net, gender_net, model_in_size, det_thres, bbox_area_thres):
        self.face_net = face_net
        self.age_net = age_net
        self.gender_net = gender_net
        self.det_thres = det_thres
        self.bbox_area_thres = bbox_area_thres
        self.FACE_MODEL_INPUT_SIZE = model_in_size  # (width, height)
        self.FACE_MODEL_MEAN_VALUES = [104, 117, 123]
        self.AGE_GENDER_INPUT_SIZE = (227, 227)
        self.AGE_GENDER_MODEL_MEAN_VALUES = (
            78.4263377603, 87.7689143744, 114.895847746)
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                         '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']


def load_net(facedet_model="weights/face_detection_caffe/res10_300x300_ssd_iter_140000.caffemodel",
             facedet_proto="weights/face_detection_caffe/deploy.prototxt.txt",
             age_proto="weights/age_net_caffe/age_deploy.prototxt",
             age_model="weights/age_net_caffe/age_net.caffemodel",
             gender_proto="weights/gender_net_caffe/gender_deploy.prototxt",
             gender_model="weights/gender_net_caffe/gender_net.caffemodel",
             model_in_size=(300, 300),
             det_thres=0.75,
             bbox_area_thres=0.10,
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
        face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")

    return Net(
        face_net, age_net, gender_net, model_in_size, det_thres, bbox_area_thres)


def inference_and_get_face_boxes(net, cv2_img):
    in_img = cv2_img.copy()
    fh, fw = in_img.shape[:2]

    in_img = pad_resize_image(in_img, net.FACE_MODEL_INPUT_SIZE)
    # network takes image in BGR format
    blob = cv2.dnn.blobFromImage(in_img, 1.0, net.FACE_MODEL_INPUT_SIZE,
                                 net.FACE_MODEL_MEAN_VALUES, True, False)
    net.face_net.setInput(blob)
    detections = net.face_net.forward()[0][0]

    h, w = cv2_img.shape[:2]
    mw, mh = net.FACE_MODEL_INPUT_SIZE
    # filter dtections below threshold
    detections = detections[detections[:, 2] > net.det_thres]
    detections[:, 3:7] = detections[:, 3:7] * np.array([mw, mh, mw, mh])

    # only select bboxes with area greater than 0.15% of total area of frame
    total_area = mw * mh
    bbox_area = ((detections[:, 5] - detections[:, 3])
                 * (detections[:, 6] - detections[:, 4]))
    bbox_area_perc = 100 * bbox_area / total_area
    detections = detections[bbox_area_perc > net.bbox_area_thres]

    # rescale detections to orig image size taking the padding into account
    boxes = detections[:, 3:7]
    boxes = scale_coords((mh, mw), boxes, (h, w)).round()
    bbox_confs = detections[:, 2]
    post_dets = PostProcessedDetection(boxes, bbox_confs=bbox_confs, bbox_areas=bbox_area_perc)
    draw_bbox_on_image(cv2_img, post_dets)
    return cv2_img, boxes


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

    frame_face, face_bboxes = inference_and_get_face_boxes(
        net, image)

    padding = 20
    for bbox in face_bboxes:
        # take face crop from face_net detections
        bbox = list(map(int, bbox))
        face = image[max(0, bbox[1] - padding):min(bbox[3] + padding, image.shape[0] - 1),
                     max(0, bbox[0] - padding):min(bbox[2] + padding, image.shape[1] - 1)]
        face_blob = cv2.dnn.blobFromImage(
            face, 1.0, net.AGE_GENDER_INPUT_SIZE, net.AGE_GENDER_MODEL_MEAN_VALUES, swapRB=False)

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
        cv2.putText(frame_face, gender_age_label, (bbox[0], bbox[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("output", frame_face)
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
        description="OpenCV DNN face detection with age and gender estimation")
    parser.add_argument("-p", "--prototxt", dest="prototxt",
                        default="weights/face_detection_caffe/deploy.prototxt.txt",
                        help="Path to 'deploy' prototxt file. (default: %(default)s)")
    parser.add_argument("--is", "--input_size", dest="input_size",
                        default=(300, 400),
                        help='Input images are resized to this (width, height). (default: %(default)s).')
    args = parser.parse_args()
    print("Current Arguments: ", args)

    # Load networks
    net = load_net(facedet_model=args.model,
                   facedet_proto=args.prototxt,
                   model_in_size=args.input_size,
                   det_thres=args.det_thres,
                   bbox_area_thres=args.bbox_area_thres,
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
