#  splits a directory with object classes in different subdirectories into
#  train, test and optionally val sub-directory with the same class sub-dir
#  structure
import os
import cv2
import sys
import glob
import logging
import mimetypes
import onnxruntime
import numpy as np
from tqdm import tqdm
from datetime import datetime

sys.path.append(".")
from modules.common_utils import get_argparse, get_file_type, fix_path_for_globbing, check_img_size, read_pickle
from modules.opencv2_dnn.utils import inference_cv2_model as inf_cv2
from modules.opencv2_dnn.utils import get_bboxes_confs_areas as get_bboxes_confs_areas_cv2
from modules.yolov5_face.onnx.onnx_utils import inference_onnx_model_yolov5_face as inf_yolov5
from modules.yolov5_face.onnx.onnx_utils import get_bboxes_confs_areas as get_bboxes_confs_areas_yolov5
from modules.openvino.utils import OVNetwork


mimetypes.init()
today = datetime.today()
year, month, day, hour, minute, sec = today.year, today.month, today.day, today.hour, today.minute, today.second

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename=f'logs/extraction_statistics_{year}{month}{day}_{hour}:{minute}:{sec}.log',
                    level=logging.INFO)

# ######################## Settings ##################################

SAVE_VIDEO_FACES_IN_SUBDIRS = True
CLASS_NAME_TO_LABEL_DICT = read_pickle("data/sample/class_name_to_label.pkl")
# max number of faces to consider from each frame for feat ext
MAX_N_FACES_PER_FRAME = 5
# max number of frames from which faces are extracted
MAX_N_FRAME_FROM_VID = 200
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
    __slots__ = ["face_net", "feature_net", "inf_func", "bbox_conf_func",
                 "feat_net_type", "det_thres", "bbox_area_thres",
                 "FACE_MODEL_MEAN_VALUES", "FACE_MODEL_INPUT_SIZE", "FACE_MODEL_OUTPUT_SIZE"]

    def __init__(self, face_net, feat_net_type,
                 inf_func, bbox_conf_func,
                 det_thres, bbox_area_thres,
                 model_in_size=(640, 640), model_out_size=None):
        self.face_net = face_net
        self.inf_func = inf_func
        self.bbox_conf_func = bbox_conf_func
        self.det_thres = det_thres
        self.bbox_area_thres = bbox_area_thres
        # in_size = (width, height), conv to int
        model_in_size = tuple(map(int, model_in_size))
        if isinstance(face_net, cv2.dnn_Net):
            self.FACE_MODEL_INPUT_SIZE = model_in_size
        else:
            # input size must be multiple of max stride 32 for yolov5 models
            self.FACE_MODEL_INPUT_SIZE = tuple(
                map(check_img_size, model_in_size))
        # only for cv2 models
        self.FACE_MODEL_MEAN_VALUES = (104.0, 117.0, 123.0)
        # (width, height), size the detected faces are resized
        # if None, no resizing is done
        self.FACE_MODEL_OUTPUT_SIZE = tuple(
            map(int, model_out_size)) if model_out_size is not None else None

        self.feat_net_type = feat_net_type
        if feat_net_type == "MOBILE_FACENET":
            self.feature_net = onnxruntime.InferenceSession(
                "weights/mobile_facenet/mobile_facenet.onnx")
        elif feat_net_type == "FACE_REID_MNV3":
            self.feature_net = OVNetwork(
                xml_path="weights/face_reidentification_retail_0095/FP32/model.xml",
                bin_path="weights/face_reidentification_retail_0095/FP32/model.bin",
                det_thres=None, bbox_area_thres=None)
        else:
            raise NotImplementedError(
                f"{feat_net_type} feature extraction net is not implemented" +
                "Supported types are ['MOBILE_FACENET', 'FACE_REID_MNV3']")

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
        feats = self.feature_net.inference_img(
            face, preprocess_func=cv2.resize)
        return feats[0].squeeze(axis=-1).squeeze(axis=-1)


class FrameFacesObj(object):

    __slots__ = ["frame_num", "time_sec", "faces", "feats", "confs", "areas"]

    def __init__(self, frame_num, time_sec, faces, feats, confs, areas):
        self.faces, self.frame_num, self.time_sec = faces, frame_num, time_sec
        self.feats, self.confs, self.areas = feats, confs, areas

    @property
    def __dict__(self):
        return {s: getattr(self, s) for s in self.__slots__ if hasattr(self, s)}


def load_net(model, prototxt, feat_net_type, det_thres, bbox_area_thres, model_in_size, model_out_size, device="cpu"):
    # load face detection model
    if device not in {"cpu", "gpu"}:
        raise NotImplementedError(f"Device {device} is not supported")

    fname, fext = os.path.splitext(model)
    if fext == ".caffemodel":
        face_net = cv2.dnn.readNetFromCaffe(prototxt, model)
    elif fext == ".pb":
        face_net = cv2.dnn.readNetFromTensorflow(model, prototxt)
    elif fext == ".onnx":
        face_net = onnxruntime.InferenceSession(model)  # ignores prototxt
    else:
        raise NotImplementedError(
            f"[ERROR] model with extension {fext} not implemented")

    if fext == ".onnx":
        inf_func = inf_yolov5
        bbox_conf_func = get_bboxes_confs_areas_yolov5
    else:
        if device == "cpu":
            face_net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU device")
        elif device == "gpu":
            face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using GPU device")
        inf_func = inf_cv2
        bbox_conf_func = get_bboxes_confs_areas_cv2
    return Net(face_net, feat_net_type,
               inf_func, bbox_conf_func,
               det_thres, bbox_area_thres,
               model_in_size, model_out_size)


def extract_face_feat_conf_area_list(net, img):
    """returns a tuple of two lists: cv2 images containing faces, conf of said face dets
    """
    if isinstance(img, str):
        image = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        image = img

    h, w = image.shape[:2]
    iw, ih = net.FACE_MODEL_INPUT_SIZE
    # mean values are ignored when yolov5 model is used
    detections = net.inf_func(
        net.face_net, image, net.FACE_MODEL_INPUT_SIZE, mean_values=net.FACE_MODEL_MEAN_VALUES)
    if detections is None:  # no faces detected
        return [], []
    # obtain bounding boxesx and conf scores
    boxes, confs, areas = net.bbox_conf_func(
        detections, net.det_thres, net.bbox_area_thres, orig_size=(w, h), in_size=(iw, ih))

    tx, ty = -6, -1
    bx, by = 4, 5

    # copy faces and feats from image
    face_list, feat_list = [], []
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box.astype('int')
        # x, y  = top x left, top y left
        # xw, yh =  bottom x right, bottom y right
        # crop face, image[ty:by, tx:bx], image[y:yh, x:xw]
        x, y, xw, yh = xmin + tx, ymin + ty, xmax + bx, ymax + by
        x, y, xw, yh = max(x, 0), max(y, 0), min(xw, w), min(yh, h)
        # .copy() only keeps crops in memory
        face = image[y:yh, x:xw].copy()
        if net.FACE_MODEL_OUTPUT_SIZE is not None:
            face = cv2.resize(face, (net.FACE_MODEL_OUTPUT_SIZE))
        feat_list.append(net.get_face_features(face.copy()))
        face_list.append(face)
    return face_list, feat_list, confs, areas


def save_extracted_faces(frames_faces_obj_list, media_root, save_dir, save_feat=True, faces_per_frame=4, feat_sz=512) -> None:
    """
    args;
        frames_faces_obj_list: list of FrameFacesObj for each frame
        media_root: root name of media file
        save_dir: dir where face imgs are saved in class dirs
    """
    target_dir = save_dir.split('/')[0]
    class_name = save_dir.split('/')[1]
    faces_savedir = os.path.join(target_dir, "faces", class_name)
    os.makedirs(faces_savedir, exist_ok=True)

    annot_dict = {"media_id": media_root, "frames_info": []}
    total = 0
    feats_list = []  # feature for one media file img/video
    for img in frames_faces_obj_list:  # for each frame
        frame_num = img.frame_num
        time_sec = img.time_sec
        prefix = '' if SAVE_VIDEO_FACES_IN_SUBDIRS else media_root + '_'
        faces, confs, areas = img.faces, img.confs, img.areas

        if save_feat:
            feats = img.feats[:faces_per_frame]
            if len(feats) < faces_per_frame:  # zero-pad if num of faces less than faces_per_frame
                feats.extend([np.zeros(feat_sz)
                              for _ in range(faces_per_frame - len(feats))])
            feats_list.extend(feats)

        single_frame_info = {"frame_num": frame_num,
                             "time_sec": time_sec, "confs": confs, "areas": areas}
        annot_dict["frames_info"].append(single_frame_info)
        i = 0
        # for each detected face
        for face, conf, area in zip(faces, confs, areas):
            i += 1
            conf = str(round(conf, 3)).replace('.', '_')
            fname = f"{prefix}_frame_{frame_num}_sec_{time_sec}_conf_{conf}_area_{area}.jpg"
            cv2.imwrite(f"{faces_savedir}/{fname}", face)
        total += i

    np_savedir = os.path.join(target_dir, "npy", class_name)
    os.makedirs(np_savedir, exist_ok=True)
    np_savepath = os.path.join(np_savedir, media_root + ".npy")
    annot_dict["class_name"] = class_name
    annot_dict["label"] = CLASS_NAME_TO_LABEL_DICT[class_name]
    if save_feat:
        annot_dict["feature"] = np.concatenate(feats_list, axis=0)
        print("ANNOT_DICT['FEATURE'] SHAPE", annot_dict["feature"].shape)
    np.save(np_savepath, annot_dict)

    return total


def filter_faces_from_data(raw_img_dir, target_dir, net):
    dir_list = glob.glob(fix_path_for_globbing(raw_img_dir))

    # for each class in raw data
    for i in tqdm(range(len(dir_list))):
        dir = dir_list[i]                # get path to class dir
        if not os.path.isdir(dir):       # skip if path is not a dir
            continue
        class_name = dir.split("/")[-1]  # get class name
        file_path_list = [file for file in glob.glob(dir + "/*")
                          if file.split(".")[-1] in VALID_FILE_EXTS]
        total_faces = 0
        # foreach image or video in file_path_list
        for media_path in file_path_list:
            # create dir for saving faces per class
            faces_save_dir = os.path.join(target_dir, class_name)

            frames_faces_obj_list = []
            media_root = os.path.basename(media_path).split('.')[0]
            mtype = get_file_type(media_path)
            if mtype == "image":
                faces, feats, confs, areas = extract_face_feat_conf_area_list(
                    net, media_path)
                frames_faces_obj_list.append(FrameFacesObj(
                    1, 1, faces, feats, confs, areas))
            elif mtype == "video":
                # save faces from videos inside sub dirs if flag is set
                if SAVE_VIDEO_FACES_IN_SUBDIRS:
                    faces_save_dir = os.path.join(faces_save_dir, media_root)
                    if os.path.exists(faces_save_dir):  # skip pre-extracted faces
                        print(
                            f"Skipping {faces_save_dir} as it already exists.")
                        continue

                cap = cv2.VideoCapture(media_path)
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
                        faces, feats, confs, areas = extract_face_feat_conf_area_list(
                            net, frame)
                        frames_faces_obj_list.append(FrameFacesObj(
                            frame_num, frame_num // step, faces, feats, confs, areas))
                    ret, frame = cap.read()
                cap.release()
                cv2.destroyAllWindows()

            faces_extracted = save_extracted_faces(
                frames_faces_obj_list, media_root, save_dir=faces_save_dir, faces_per_frame=MAX_N_FACES_PER_FRAME)
            total_faces += faces_extracted
        logging.info(f"{total_faces} faces extracted for class {class_name}")


def main():
    parser = get_argparse(description="Dataset face extraction")
    parser.remove_argument("input_src")
    parser.add_argument('-rd', '--raw_datadir_path',
                        type=str, required=True,
                        help="""Raw dataset dir path with
                        class imgs inside folders.""")
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
    parser.add_argument("-os", "--output_size",
                        nargs=2,
                        help='Output face images are resized to this (width, height) -os 112 112. (default: %(default)s).')
    args = parser.parse_args()
    print("Current Arguments: ", args)
    net = load_net(model=args.model,
                   prototxt=args.prototxt,
                   feat_net_type=args.face_feat_type,
                   det_thres=args.det_thres,
                   bbox_area_thres=args.bbox_area_thres,
                   model_in_size=args.input_size,
                   model_out_size=args.output_size,
                   device=args.device)

    filter_faces_from_data(args.raw_datadir_path,
                           args.target_datadir_path,
                           net)


if __name__ == "__main__":
    main()
