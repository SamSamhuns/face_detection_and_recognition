import os
import sys
import cv2
import glob
import logging
import onnxruntime
import numpy as np
from tqdm import tqdm
from datetime import datetime

sys.path.append(".")
from modules.utils.parser import get_argparse
from modules.utils.files import get_file_type
from modules.utils.image import calculate_bbox_iou, get_distinct_rgb_color
from modules.yolov5_face.onnx.onnx_utils import check_img_size
from modules.yolov5_face.onnx.onnx_utils import inference_onnx_model_yolov5_face
from modules.yolov5_face.onnx.onnx_utils import get_bboxes_confs_areas as get_bboxes_confs_areas_yolov5
from modules.openvino.model import OVFeatModel

today = datetime.today()
year, month, day, hour, minute, sec = today.year, today.month, today.day, today.hour, today.minute, today.second

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename=f'logs/extraction_and_label_statistics_{year}{month}{day}_{hour}:{minute}:{sec}.log',
                    level=logging.INFO)

# ######################## Settings ##################################

ROOT_URL = "https://example/"
CV2_DISP_WAIT_MS = 100
# X & Y displacements for cv2 window
CV2_DISP_LOC_X = 600
CV2_DISP_LOC_Y = 50
MAX_N_FRAME_FROM_VID = 200  # max number of frames from which faces are extracted
VALID_FILE_EXTS = {'jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm',
                   'mp4', 'avi'}

# #################### Raw Data Organization #########################
#        dataset
#              |_ class_1
#                        |_ img1/vid1
#                        |_ img2/vid2
#                        |_ ....
#              |_ class_2
#                        |_ img1/vid1
#                        |_ img2/vid2
#                        |_ ....
#              ...
#
# example raw data path    = "raw_data/dataset"
# example target data path = "target_data/dataset"
# ###################################################################


class Net(object):
    __slots__ = ["face_net", "feature_net",
                 "inf_func", "bbox_conf_func",
                 "det_thres", "bbox_area_thres",
                 "FACE_MODEL_MEAN_VALUES", "FACE_MODEL_INPUT_SIZE",
                 "feat_net_type", "face_feat_bbox_age_gender_list",
                 "normal_thres", "harsh_thres", "use_bbox_iou", "max_faceid"]

    def __init__(self, face_net, feat_net_type,
                 inf_func, bbox_conf_func,
                 det_thres, bbox_area_thres,
                 model_in_size=(640, 640),
                 use_bbox_iou_to_track_face=True):
        self.face_net = face_net
        self.inf_func = inf_func
        self.bbox_conf_func = bbox_conf_func
        self.det_thres = det_thres
        self.bbox_area_thres = bbox_area_thres
        # in_size = (width, height), conv to int
        model_in_size = tuple(map(int, model_in_size))
        # input size must be multiple of max stride 32 for yolov5 models
        self.FACE_MODEL_INPUT_SIZE = tuple(map(check_img_size, model_in_size))
        # only for cv2 models
        self.FACE_MODEL_MEAN_VALUES = (104.0, 117.0, 123.0)

        # ################## parameters for face tracking #################### #
        self.face_feat_bbox_age_gender_list = []
        self.normal_thres = 1.
        self.harsh_thres = 0.72
        self.use_bbox_iou = use_bbox_iou_to_track_face
        self.max_faceid = 0  # to track number of uniq faces

        self.feat_net_type = feat_net_type
        if feat_net_type == "MOBILE_FACENET":
            self.feature_net = onnxruntime.InferenceSession(
                "weights/mobile_facenet/mobile_facenet.onnx")
        elif feat_net_type == "FACE_REID_MNV3":
            self.feature_net = OVFeatModel(
                xml_path="weights/face_reidentification_retail_0095/FP32/model.xml",
                bin_path="weights/face_reidentification_retail_0095/FP32/model.bin",
                verbose=False)
        else:
            raise NotImplementedError(
                f"{feat_net_type} feature extraction net is not implemented."
                "Supported types are ['MOBILE_FACENET', 'FACE_REID_MNV3']")

    def check_if_face_exists(self, new_feat, new_bbox):
        for i, (faceid, feat, bbox, age, gender) in enumerate(self.face_feat_bbox_age_gender_list):
            if self.feat_net_type == "MOBILE_FACENET":
                dist = np.linalg.norm(feat - new_feat)
            else:
                dist = 1 - (np.inner(feat, new_feat) / (np.linalg.norm(feat) * np.linalg.norm(new_feat)))
            if self.use_bbox_iou:
                iou = calculate_bbox_iou(bbox, new_bbox)
            print(f"dist={dist:.3f}, bbox iou={iou:.3f}")
            if (dist < self.normal_thres and iou > 0.1) or dist < self.harsh_thres:
                print(
                    f"Same face detected: iou={iou:.3f} and dist={dist:.3f}")
                self.face_feat_bbox_age_gender_list[i][1] = new_feat
                self.face_feat_bbox_age_gender_list[i][2] = new_bbox
                return True, faceid, age, gender
        return False, None, None, None

    def add_face(self, feat, bbox, age, gender):
        self.max_faceid += 1
        self.face_feat_bbox_age_gender_list.append(
            [self.max_faceid, feat, bbox, age, gender])

    def clear_faces(self):
        print("Clearing existing faces from face tracker.")
        self.max_faceid = 0
        self.face_feat_bbox_age_gender_list = []

    def get_num_unique_faces(self):
        return self.max_faceid

    def get_face_features(self, face):
        if self.feat_net_type == "MOBILE_FACENET":
            feats = self._get_face_features_mobile_facenet(face)
        elif self.feat_net_type == "FACE_REID_MNV3":
            feats = self._get_face_features_face_reid(face)
        return feats

    def _get_face_features_mobile_facenet(self, face):
        """
        Get face features with mobile_facenet
        """
        # 112, 112 is the input size of mobile_facenet
        face = (cv2.resize(face, (112, 112)) - 127.5) / 127.5
        # HWC to BCHW
        face = np.expand_dims(np.transpose(face, (2, 0, 1)),
                              axis=0).astype(np.float32)
        features = self.feature_net.run(None, {"images": face})  # BGR fmt
        return features[0][0]

    def _get_face_features_face_reid(self, face):
        """
        Get face features with face re-identification MobileNet-V2 model
        """
        feats = self.feature_net(face).astype(np.float32)
        return feats


class FrameFacesObj(object):

    __slots__ = ["faces", "ids", "frame_num", "time_sec",
                 "bboxes", "confs", "ages", "genders"]

    def __init__(self, faces, ids, frame_num, time_sec, bboxes, confs, ages, genders):
        self.faces, self.ids = faces, ids
        self.frame_num, self.time_sec = frame_num, time_sec
        self.bboxes, self.confs = bboxes, confs
        self.ages, self.genders = ages, genders

    @property
    def __dict__(self):
        return {s: getattr(self, s) for s in self.__slots__ if hasattr(self, s)}


def load_net(model, feat_net_type, det_thres, bbox_area_thres, model_in_size, device="cpu"):
    # load face detection model
    if device not in {"cpu", "gpu"}:
        raise NotImplementedError(f"Device {device} is not supported")

    fname, fext = os.path.splitext(model)
    if fext == ".onnx":
        face_net = onnxruntime.InferenceSession(model)  # ignores prototxt
    else:
        raise NotImplementedError(
            f"[ERROR] model with extension {fext} not implemented")

    if fext == ".onnx":
        inf_func = inference_onnx_model_yolov5_face
        bbox_conf_func = get_bboxes_confs_areas_yolov5

    return Net(face_net, feat_net_type, inf_func, bbox_conf_func,
               det_thres, bbox_area_thres, model_in_size)


def get_age_and_gender_with_cv2_waitKey(image):
    gender, age = '', ''
    age_groups = ['0-5', '5-12', '12-20', '20-50', '50-100']

    gender_label = "w: male & e: female"
    h, w = image.shape[:2]
    image_gender = image.copy()
    tl = round(0.002 * (h + w) / 2) + 1
    cv2.rectangle(image_gender, (0, h - 40), (w, h),
                  (0, 0, 0), cv2.FILLED, cv2.LINE_AA)
    cv2.putText(image_gender, gender_label, (5, h - 5), 0, tl / 2, [0, 0, 255],
                thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow("Face Detection", image_gender)
    cv2.moveWindow('Face Detection', CV2_DISP_LOC_X, CV2_DISP_LOC_Y)
    while True:
        key = cv2.waitKey(-1)
        if key == 119:       # w pressed for male
            gender = "male"
            break
        elif key == 101:     # e pressed for female
            gender = "female"
            break

    age_label1 = "Age 1: 0-5   2: 5-12  3: 12-20"
    age_label2 = "    4: 20-50 5: 50-100"
    image_age = image.copy()
    tl = round(0.002 * (h + w) / 2) + 1
    cv2.rectangle(image_age, (0, h - 60), (w, h),
                  (0, 0, 0), cv2.FILLED, cv2.LINE_AA)
    cv2.putText(image_age, age_label1, (5, h - 35), 0, tl / 3, [0, 0, 255],
                thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(image_age, age_label2, (5, h - 5), 0, tl / 3, [0, 0, 255],
                thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow("Face Detection", image_age)
    cv2.moveWindow('Face Detection', CV2_DISP_LOC_X, CV2_DISP_LOC_Y)
    while True:
        key = cv2.waitKey(-1)
        if 49 <= key <= 53:  # 1, 2, 3, 4, or 5 pressed for age groups
            age = age_groups[key - 49]
            break
    return age, gender


def extract_face_img_id_bbox_conf_age_gender_list(net, img):
    """returns a tuple of six lists: cv2 face images, ids, bounding boxes, confidences, ages, & genders
    """
    if isinstance(img, str):
        orig_image = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        orig_image = img

    h, w = orig_image.shape[:2]
    iw, ih = net.FACE_MODEL_INPUT_SIZE
    faces, bboxes, faceids, confs, ages, genders = [], [], [], [], [], []

    # pass the blob through the network to get raw detections
    detections = net.inf_func(net.face_net, orig_image,
                              net.FACE_MODEL_INPUT_SIZE)
    if detections is None:  # no faces detected
        return faces, faceids, bboxes, confs, ages, genders
    # obtain bounding boxesx and conf scores
    boxes, confs, areas = net.bbox_conf_func(
        detections, net.det_thres, net.bbox_area_thres, orig_size=(w, h), in_size=(iw, ih))
    tx, ty = -10, -1
    bx, by = 10, 5

    # copy faces and face features from image
    tl = round(0.002 * (w + h) / 2) + 1
    for i, box in enumerate(boxes):
        # use a new fresh image to show which face is being annotated
        image = orig_image.copy()
        xmin, ymin, xmax, ymax = box.astype('int')
        # x, y  = top x left, top y left
        # xw, yh =  bottom x right, bottom y right
        # crop face, image[ty:by, tx:bx], image[y:yh, x:xw]
        x, y, xw, yh = xmin + tx, ymin + ty, xmax + bx, ymax + by
        x, y, xw, yh = max(x, 0), max(y, 0), min(xw, w), min(yh, h)
        bboxes.append([x, y, xw, yh])

        # .copy() only keeps crops in memory
        face = image[y:yh, x:xw].copy()
        faces.append(face)

        # extract face features for tracking
        waitKey_value = CV2_DISP_WAIT_MS
        face_feat = net.get_face_features(face.copy())

        exists, faceid, age, gender = net.check_if_face_exists(
            face_feat, (x, y, xw, yh))
        # if face does not exist, new faceid is assigned
        faceid = net.max_faceid + 1 if not exists else faceid

        # draw face bbox and conf label on image
        cv2.rectangle(image, (x, y), (xw, yh), get_distinct_rgb_color(faceid), thickness=max(
            int((w + h) / 600), 2), lineType=cv2.LINE_AA)
        label = f"ID:{faceid}_{confs[i]:.2f}"
        t_size = cv2.getTextSize(
            label, 0, fontScale=tl / 3, thickness=1)[0]
        c2 = x + t_size[0] + 3, y - t_size[1] - 5
        cv2.rectangle(image, (x - 1, y), c2,
                      (0, 0, 0), cv2.FILLED, cv2.LINE_AA)
        cv2.putText(image, label, (x + 3, y - 4), 0, tl / 3, [255, 255, 255],
                    thickness=1, lineType=cv2.LINE_AA)

        if exists:
            # display image with detections
            cv2.imshow("Face Detection", image)
            cv2.moveWindow('Face Detection', CV2_DISP_LOC_X, CV2_DISP_LOC_Y)
            print("Existing face recognized. Using existing age and gender")
        else:
            # display image with detections and instructions for entering age and gender
            print("New face detected. Set gender and age")
            age, gender = get_age_and_gender_with_cv2_waitKey(image)
            net.add_face(face_feat, (x, y, xw, yh), age, gender)
        faceids.append(faceid)
        ages.append(age)
        genders.append(gender)
        cv2.waitKey(waitKey_value)

    return faces, faceids, bboxes, confs, ages, genders


def save_extracted_faces(frames_faces_obj_list, media_root, class_name, save_face, faces_save_dir, save_feat, feats_save_dir):
    """
    Save extracted faces for one image or one video
    args:
        frames_faces_obj_list: list of FrameFacesObj for each frame
        media_root: root name of media file
        faces_save_dir: dir where face imgs are saved in class dirs
    """
    if save_face:
        os.makedirs(faces_save_dir, exist_ok=True)

    annot_dict = {"media_id": media_root, "frames_info": []}
    total = 0
    for img in frames_faces_obj_list:  # for each frame
        frame_num = img.frame_num
        time_sec = img.time_sec
        faces, ids, bboxes, confs, ages, genders = (
            img.faces, img.ids, img.bboxes, img.confs, img.ages, img.genders)

        single_frame_info = {"frame_num": frame_num, "time_sec": time_sec,
                             "face_ids": ids, "face_bboxes": bboxes, "confs": confs,
                             "ages": ages, "genders": genders}
        annot_dict["frames_info"].append(single_frame_info)
        i = 0
        # for each detected face
        for face, id, bbox, conf, age, gender in zip(faces, ids, bboxes, confs, ages, genders):
            i += 1
            if save_face:
                conf = str(round(conf, 3)).replace('.', '_')
                fname = f"frame_{frame_num}_sec_{time_sec}_id_{id}_conf_{conf}_{gender}_{age}.jpg"
                cv2.imwrite(f"{faces_save_dir}/{fname}", face)
        total += i

    os.makedirs(feats_save_dir, exist_ok=True)
    npy_savepath = os.path.join(feats_save_dir, media_root + ".npy")
    annot_dict["class_name"] = class_name
    annot_dict["media_url"] = ROOT_URL + media_root
    np.save(npy_savepath, annot_dict)

    return total


def filter_faces_from_data(source_dir, target_dir, net):
    class_dir_list = glob.glob(os.path.join(source_dir, "*"))

    total_media_ext, total_faces_ext = 0, 0
    # for each class in raw data
    for i in tqdm(range(len(class_dir_list))):
        class_dir = class_dir_list[i]          # get path to class dir
        if not os.path.isdir(class_dir):       # skip if path is not a dir
            continue
        class_name = class_dir.split("/")[-1]
        file_path_list = [file for file in glob.glob(class_dir + "/*")
                          if file.split(".")[-1] in VALID_FILE_EXTS]

        class_media_ext, class_faces_ext = 0, 0
        # foreach image or video in file_path_list
        for media_path in file_path_list:
            try:
                # create dir for saving faces/feats per class
                faces_save_dir = os.path.join(
                    target_dir, "faces", class_name)
                feats_save_dir = os.path.join(
                    target_dir, "npy_feat", class_name)

                frames_faces_obj_list = []
                media_root = os.path.basename(media_path).split('.')[0]
                mtype = get_file_type(media_path)
                if mtype == "image":
                    # Note: for images, face feature extraction and tracking is not implemented
                    faces, faceids, bboxes, confs, ages, genders = extract_face_img_id_bbox_conf_age_gender_list(
                        net, media_path)
                    frames_faces_obj_list.append(FrameFacesObj(
                        faces, faceids, 1, 1, bboxes, confs, ages, genders))
                elif mtype == "video":
                    faces_save_dir = os.path.join(faces_save_dir, media_root)
                    if os.path.exists(faces_save_dir):  # skip pre-extracted faces
                        print(
                            f"Skipping {faces_save_dir} as it already exists.")
                        continue

                    cap = cv2.VideoCapture(media_path)
                    # take 1 frame per sec
                    step = int(round(cap.get(cv2.CAP_PROP_FPS)))
                    frame_num = 0
                    save_frames_num = 0
                    ret, frame = cap.read()
                    while ret:
                        frame_num += 1
                        if frame_num % step == 0 or frame_num == 1:
                            save_frames_num += 1
                            if save_frames_num > MAX_N_FRAME_FROM_VID:
                                break
                            faces, faceids, bboxes, confs, ages, genders = extract_face_img_id_bbox_conf_age_gender_list(
                                net, frame)
                            frames_faces_obj_list.append(FrameFacesObj(
                                faces, faceids, frame_num, frame_num // step, bboxes, confs, ages, genders))
                        ret, frame = cap.read()
                    cap.release()
                    cv2.destroyAllWindows()
                    net.clear_faces()

                faces_extracted = save_extracted_faces(
                    frames_faces_obj_list, media_root, class_name, True, faces_save_dir, True, feats_save_dir)
                class_faces_ext += faces_extracted
                class_media_ext += 1
            except Exception as e:
                print(f"{e}. Extraction failed for media {media_path}")
        total_faces_ext += class_faces_ext
        total_media_ext += class_media_ext
        logging.info(
            f"{class_faces_ext} faces found for class {class_name} in {class_media_ext} files")
    logging.info(
        f"{total_faces_ext} faces extracted from {total_media_ext} files")


def main():
    parser = get_argparse(
        description="Dataset face extraction & labelling", conflict_handler='resolve')
    parser.remove_arguments(["input_src", "prototxt"])
    parser.add_argument("-m", "--model",
                        default="weights/yolov5s/yolov5s-face.onnx",
                        help='Path to weight file (.pth/.onnx). (default: %(default)s).')
    parser.add_argument('-sd', '--source_datadir_path',
                        type=str, required=True,
                        help="""Source dataset dir path with
                        class imgs/vids inside folders.""")
    parser.add_argument('-td', '--target_datadir_path',
                        type=str, default="face_data",
                        help="""Target dataset dir path where
                        imgs will be sep into train & test. (default: %(default)s)""")
    parser.add_argument("-ft", "--face_feat_type", default="FACE_REID_MNV3",
                        choices=["FACE_REID_MNV3", "MOBILE_FACENET"],
                        help="Type of face feature extracter to use for tracking. (default: %(default)s)")
    parser.add_argument("-is", "--input_size",
                        nargs=2,
                        default=(300, 400),
                        help='Input images are resized to this (width, height) -is 300 400. (default: %(default)s).')
    args = parser.parse_args()
    logging.info(f"Arguments used: {args}")
    print("Current Arguments: ", args)

    net = load_net(model=args.model,
                   feat_net_type=args.face_feat_type,
                   det_thres=args.det_thres,
                   bbox_area_thres=args.bbox_area_thres,
                   model_in_size=args.input_size,
                   device=args.device)

    filter_faces_from_data(args.source_datadir_path,
                           args.target_datadir_path,
                           net)


if __name__ == "__main__":
    main()
