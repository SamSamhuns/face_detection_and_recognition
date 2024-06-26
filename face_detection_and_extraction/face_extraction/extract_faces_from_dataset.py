import os
import cv2
import sys
import glob
import time
import torch
import logging
import mimetypes
import traceback
import onnxruntime
import numpy as np
from tqdm import tqdm
from datetime import datetime

sys.path.append(".")
from modules.utils.parser import get_argparse
from modules.utils.files import get_file_type, read_json, gen_class2label_from_dir
from modules.utils.image import check_img_size, scale_coords
from modules.opencv2_dnn.model import OpenCVFaceDetModel
from modules.yolov5_face.onnx.onnx_utils import inference_onnx_model_yolov5_face as inf_yolov5
from modules.yolov5_face.onnx.onnx_utils import get_bboxes_confs_areas as get_bboxes_confs_areas_yolov5
from modules.face_detection_trt_server.inference import TritonServerInferenceSession as face_det_trt_sess
from modules.facenet_trt_server.inference import TritonServerInferenceSession as face_feat_trt_sess
from modules.facenet_age_trt_server.inference import TritonServerInferenceSession as face_age_trt_sess
from modules.facenet_gender_trt_server.inference import TritonServerInferenceSession as face_gender_trt_sess
from modules.openvino.model import OVFeatModel


today = datetime.today()
year, month, day, hour, minute, sec = today.year, today.month, today.day, today.hour, today.minute, today.second

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename=f'logs/extraction_statistics_{year}{month}{day}_{hour}:{minute}:{sec}.log',
                    level=logging.INFO)

# ######################## Settings ##################################
# max number of faces to consider from each frame for feat ext
MAX_N_FACES_PER_FRAME = 3
# max number of frames from which faces are extracted
MAX_N_FRAME_FROM_VID = 15
VALID_FILE_EXTS = {'jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm',
                   'mp4', 'avi'}

# ######################## Raw Data Organization ###############################
#        dataset
#              |_ class_1
#                        |_ {img1/vid1}
#                        |_ {img2/vid2}
#                        |_ ....
#              |_ class_2
#                        |_ {img1/vid1}
#                        |_ {img2/vid2}
#                        |_ ....
#              ...
# ##############################################################################

mimetypes.init()


class Net(object):
    __slots__ = ["face_net", "feature_net", "inf_func", "bbox_conf_area_func",
                 "feat_net_type", "det_thres", "bbox_area_thres",
                 "face_age_net", "face_gender_net",
                 "FACE_MODEL_MEAN_VALUES",
                 "FACE_MODEL_INPUT_SIZE",
                 "FACE_FEATURE_SIZE"]

    def __init__(self, face_net, feat_net_type,
                 inf_func, bbox_conf_area_func,
                 det_thres, bbox_area_thres,
                 model_in_size=(640, 640)):
        self.face_net = face_net
        self.inf_func = inf_func
        self.bbox_conf_area_func = bbox_conf_area_func
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

        self.feat_net_type = feat_net_type
        if feat_net_type == "MOBILE_FACENET_ONNX":
            self.feature_net = onnxruntime.InferenceSession(
                "weights/mobile_facenet/mobile_facenet.onnx")
            self.FACE_FEATURE_SIZE = 512
        elif feat_net_type == "FACE_REID_MNV2":
            self.feature_net = OVFeatModel(
                xml_path="weights/face_reidentification_retail_0095/FP32/model.xml",
                bin_path="weights/face_reidentification_retail_0095/FP32/model.bin",
                verbose=False)
            self.FACE_FEATURE_SIZE = 256
        elif feat_net_type == "FACENET_OV":
            self.feature_net = OVFeatModel(
                xml_path="weights/facenet_20180408_102900/facenet_openvino/20180408-102900.xml",
                bin_path="weights/facenet_20180408_102900/facenet_openvino/20180408-102900.bin",
                verbose=False)
            self.FACE_FEATURE_SIZE = 512
        elif feat_net_type == "FACENET_TRT":
            self.feature_net = face_feat_trt_sess()
            self.FACE_FEATURE_SIZE = 128
        elif feat_net_type == "FACENET_AGE_GENDER":
            self.face_age_net = face_age_trt_sess()
            self.face_gender_net = face_gender_trt_sess()
            self.FACE_FEATURE_SIZE = 6
        elif feat_net_type == "CAFFE_AGE_GENDER":
            # TODO complete this
            self.face_age_net = None
            self.face_gender_net = None
            self.FACE_FEATURE_SIZE = 10
        else:
            raise NotImplementedError(
                f"""{feat_net_type} feature extraction net is not implemented""")

    def get_face_features(self, face):
        if self.feat_net_type == "MOBILE_FACENET_ONNX":   # out shape 512
            feats = self._get_face_features_mobile_facenet_onnx(face)
        elif self.feat_net_type == "FACE_REID_MNV2":      # out shape 256
            feats = self._get_face_features_openvino(face)
        elif self.feat_net_type == "FACENET_OV":          # out shape 512
            feats = self._get_face_features_openvino(face)
        elif self.feat_net_type == "FACENET_TRT":         # out shape 128
            feats = self._get_face_features_trt(face)
        elif self.feat_net_type == "FACENET_AGE_GENDER":  # out shape 6 (2+4)
            feats = self._get_face_age_gender_trt(face)
        # TODO add CAFFE_AGE_GENDER
        return feats

    def _get_face_age_gender_trt(self, face):
        """
        Get face age and gender concatenated features with triton server inference
        """
        pred_age = self.face_age_net.inference_trt_model_facenet(
            self.face_age_net, face, (160, 160))
        pred_gender = self.face_gender_net.inference_trt_model_facenet(
            self.face_gender_net, face, (160, 160))
        # feats = np.asarray([np.argmax(pred_age), np.argmax(pred_gender)])
        feats = np.concatenate([pred_age, pred_gender], axis=None)
        return feats

    def _get_face_features_trt(self, face):
        """
        Get face features with triton server inference
        """
        features = self.feature_net.inference_trt_model_facenet(
            self.feature_net, face, (160, 160))
        return features

    def _get_face_features_mobile_facenet_onnx(self, face):
        """
        Get face features with MOBILE_FACENET_ONNX
        """
        # 112, 112 is the input size of MOBILE_FACENET_ONNX
        face = (cv2.resize(face, (112, 112)) - 127.5) / 127.5
        # HWC to BCHW
        face = np.expand_dims(np.transpose(face, (2, 0, 1)),
                              axis=0).astype(np.float32)
        features = self.feature_net.run(None, {"images": face})  # BGR fmt
        return features[0][0]

    def _get_face_features_openvino(self, face):
        """
        Get face features with openvino face feat ext model
        i.e. face re-identification MobileNet-V2, facenet_20180408_102900
        """
        features = self.feature_net(face).astype(np.float32)
        return features


class FrameFacesObj(object):

    __slots__ = ["frame_num", "time_sec", "faces", "feats", "confs", "areas"]

    def __init__(self, frame_num, time_sec, faces, feats, confs, areas):
        self.faces, self.frame_num, self.time_sec = faces, frame_num, time_sec
        self.feats, self.confs, self.areas = feats, confs, areas

    @property
    def __dict__(self):
        return {s: getattr(self, s) for s in self.__slots__ if hasattr(self, s)}


def load_net(model, prototxt, feat_net_type, det_thres, bbox_area_thres, model_in_size, device="cpu"):
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
    elif fext in {".pt", ".pth"}:
        sys.path.append("modules/yolov5_face/pytorch")
        from modules.yolov5_face.pytorch import attempt_load
        device = torch.device("cuda" if device == "gpu" and torch.cuda.is_available() else "cpu")
        face_net = attempt_load(model, device)
    elif fname == "modules/face_detection_trt_server":
        face_net = face_det_trt_sess(det_thres, bbox_area_thres, device=device)
    else:
        raise NotImplementedError(
            f"[ERROR] model with extension {fext} not implemented")

    if fext == ".onnx":
        inf_func = inf_yolov5
        bbox_conf_area_func = get_bboxes_confs_areas_yolov5
    elif fext in {".pt", ".pth"}:
        from modules.yolov5_face.pytorch import inference_pytorch_model_yolov5_face as inf_yolov5_pt

        def inf_func(net, image, *args, **kwargs):
            return inf_yolov5_pt(net, image, *args)
        bbox_conf_area_func = get_bboxes_confs_areas_yolov5
    elif fname == "modules/face_detection_trt_server":
        inf_func = face_net.inference_trt_model_yolov5_face
        bbox_conf_area_func = get_bboxes_confs_areas_yolov5
    else:
        if device == "cpu":
            face_net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU device")
        elif device == "gpu":
            face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using GPU device")

        opencv2_model = OpenCVFaceDetModel(face_net, model_in_size, det_thres, bbox_area_thres)

        def inf_func(net, image, *args, **kwargs):
            return opencv2_model(image)

        def bbox_conf_area_func(dets, det_thres, bbox_area_thres, orig_size, in_size):
            w, h = orig_size
            iw, ih = in_size

            # filter dets below threshold
            dets = dets[dets[:, -1] > det_thres]
            # denorm bounding boxes and optional landmark coords to model input_size
            dets[:, :-1] = dets[:, :-1] * np.array([iw, ih] * ((dets.shape[-1] - 1) // 2))
            # only select bboxes with area greater than bbox_area_thres of total area of frame
            total_area = iw * ih
            bbox_area = ((dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1]))
            bbox_area_perc = bbox_area / total_area
            bbox_area_perc_filter = (100 * bbox_area_perc) > bbox_area_thres
            dets = dets[bbox_area_perc_filter]
            # select bbox_area_percs higher than bbox_area_thres
            bbox_area_perc = bbox_area_perc[bbox_area_perc_filter]
            areas = bbox_area_perc

            confs = dets[:, -1]
            dets = dets[:, :-1]  # discard bbox conf scores
            # rescale dets to orig image size taking the padding into account
            dets = scale_coords((ih, iw), dets, (h, w)).round()
            # add bbox coords
            boxes = dets[:, :4]
            return boxes, confs, areas

    return Net(face_net, feat_net_type,
               inf_func, bbox_conf_area_func,
               det_thres, bbox_area_thres,
               model_in_size)


def extract_face_feat_conf_area_list(net, img, save_feat):
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
        return [], [], [], []
    # obtain bounding boxe coords, conf scores and areas
    boxes, confs, areas = net.bbox_conf_area_func(
        detections, net.det_thres, net.bbox_area_thres, orig_size=(w, h), in_size=(iw, ih))

    # face bbox offsets
    tx, ty = -6, -1
    bx, by = 4, 5

    # copy faces and feats from image
    face_list, feat_list = [], []
    for box in boxes:
        xmin, ymin, xmax, ymax = box.astype('int')
        # x, y  = top x left, top y left
        # xw, yh =  bottom x right, bottom y right
        # crop face, image[ty:by, tx:bx], image[y:yh, x:xw]
        x, y, xw, yh = xmin + tx, ymin + ty, xmax + bx, ymax + by
        x, y, xw, yh = max(x, 0), max(y, 0), min(xw, w), min(yh, h)
        # .copy() only keeps crops in memory
        face = image[y:yh, x:xw].copy()
        if save_feat:
            feat_list.append(net.get_face_features(face.copy()))
        face_list.append(face)
    return face_list, feat_list, confs, areas


def save_extracted_faces(frames_faces_obj_list, media_root, class_name,
                         save_face, faces_save_dir,
                         save_feat, feats_save_dir, face_feature_size,
                         class2label_dict):
    """
    args;
        frames_faces_obj_list: list of FrameFacesObj for each frame
        media_root: root name of media file
        save_dir: dir where face imgs are saved in class dirs
    """
    if save_face:
        os.makedirs(faces_save_dir, exist_ok=True)
    annot_dict = {"media_id": media_root, "frames_info": []}
    total = 0
    feats_list = []  # feature for one media file img/video
    for img in frames_faces_obj_list:  # for each frame
        frame_num = img.frame_num
        time_sec = img.time_sec
        faces, confs, areas = img.faces, img.confs, img.areas

        if save_feat:
            feats = img.feats[:MAX_N_FACES_PER_FRAME]
            # zero-pad if num of faces less than faces_per_frame
            if len(feats) < MAX_N_FACES_PER_FRAME:
                face_diff = MAX_N_FACES_PER_FRAME - len(feats)
                feats.extend([np.zeros(face_feature_size)
                              for _ in range(face_diff)])
            feats_list.extend(feats)

        single_frame_info = {"frame_num": frame_num,
                             "time_sec": time_sec, "confs": confs, "areas": areas}
        annot_dict["frames_info"].append(single_frame_info)
        i = 0
        # for each detected face
        for face, conf, area in zip(faces, confs, areas):
            i += 1
            if save_face:
                conf = str(round(conf, 3)).replace('.', '_')
                fname = f"frame_{frame_num}_sec_{time_sec}_conf_{conf}_area_{area}.jpg"
                cv2.imwrite(f"{faces_save_dir}/{fname}", face)
        total += i

    os.makedirs(feats_save_dir, exist_ok=True)
    npy_savepath = os.path.join(feats_save_dir, media_root + ".npy")
    annot_dict["class_name"] = class_name
    annot_dict["label"] = class2label_dict[class_name]
    if save_feat:
        if len(frames_faces_obj_list) < MAX_N_FRAME_FROM_VID:
            frame_diff = MAX_N_FRAME_FROM_VID - len(frames_faces_obj_list)
            feats_list.extend([np.zeros(face_feature_size)
                               for _ in range(MAX_N_FACES_PER_FRAME)] * frame_diff)
        annot_dict["feature"] = np.concatenate(
            feats_list, axis=0).astype(np.float32)
    np.save(npy_savepath, annot_dict)

    return total


def filter_faces_from_data(source_dir, target_dir, net, save_face, save_feat):
    init_tm = time.time()
    dir_list = glob.glob(os.path.join(source_dir, "*"))

    json_label_path = os.path.join(source_dir, "class2label.json")
    gen_class2label_from_dir(source_dir, json_label_path)
    class2label_dict = read_json(json_label_path)

    total_media_ext, total_faces_ext = 0, 0
    skip_class = set([])
    # for each class in raw data
    for i in tqdm(range(len(dir_list))):
        dir = dir_list[i]                # get path to class dir
        if not os.path.isdir(dir):       # skip if path is not a dir
            continue
        class_name = dir.split("/")[-1]  # get class name
        if class_name in skip_class:
            continue
        print(f"Faces will be extracted from class {class_name}")
        file_path_list = [file for file in glob.glob(dir + "/*")
                          if file.split(".")[-1] in VALID_FILE_EXTS]

        class_media_ext, class_faces_ext = 0, 0
        # foreach image or video in file_path_list
        for media_path in file_path_list:
            try:
                # create dir for saving faces/feats per class
                faces_save_dir = os.path.join(
                    target_dir, "faces", class_name)
                feats_save_dir = os.path.join(
                    target_dir, f"npy_feat_{net.FACE_FEATURE_SIZE}", class_name)

                frames_faces_obj_list = []
                media_root = os.path.basename(media_path).split('.')[0]
                mtype = get_file_type(media_path)
                if mtype == "image":
                    faces, feats, confs, areas = extract_face_feat_conf_area_list(
                        net, media_path, save_feat)
                    frames_faces_obj_list.append(FrameFacesObj(
                        1, 1, faces, feats, confs, areas))
                elif mtype == "video":
                    faces_save_dir = os.path.join(faces_save_dir, media_root)
                    if os.path.exists(faces_save_dir):  # skip pre-extracted faces
                        print(
                            f"Skipping {faces_save_dir} as it already exists.")
                        continue
                    feats_save_path = os.path.join(
                        feats_save_dir, media_root + ".npy")
                    if os.path.exists(feats_save_path):  # skip pre-extracted feats
                        print(
                            f"Skipping {feats_save_path} as it already exists.")
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
                                net, frame, save_feat)
                            frames_faces_obj_list.append(FrameFacesObj(
                                frame_num, frame_num // step, faces, feats, confs, areas))
                        ret, frame = cap.read()
                    cap.release()
                    cv2.destroyAllWindows()

                faces_extracted = save_extracted_faces(
                    frames_faces_obj_list, media_root, class_name, save_face,
                    faces_save_dir, save_feat, feats_save_dir, net.FACE_FEATURE_SIZE,
                    class2label_dict)
                class_faces_ext += faces_extracted
                class_media_ext += 1
            except Exception as e:
                print(f"{e}. Extraction failed for media {media_path}")
                traceback.print_exc()
        total_faces_ext += class_faces_ext
        total_media_ext += class_media_ext
        logging.info(
            f"{class_faces_ext} faces found for class {class_name} in {class_media_ext} files")
    logging.info(
        f"{total_faces_ext} faces extracted from {total_media_ext} files")
    logging.info(
        f"Total time taken: {time.time() - init_tm:.2f}s")
    print(
        f"Time for extracting {total_faces_ext} faces is {time.time() - init_tm:.2f}s")
    if isinstance(net.face_net, face_det_trt_sess):
        print("Shutting down triton-server docker container")
        net.face_net.container.kill()


def main():
    parser = get_argparse(description="Dataset face extraction")
    parser.remove_argument("input_src")
    parser.add_argument('--sd', '--source_datadir_path', dest="source_datadir_path",
                        type=str, required=True,
                        help="""Source dataset dir path with
                        class imgs/vids inside folders.""")
    parser.add_argument('--td', '--target_datadir_path', dest="target_datadir_path",
                        type=str, default="face_data",
                        help="""Target dataset dir path where
                        imgs will be sep into train & test. (default: %(default)s)""")
    parser.add_argument("--ft", "--face_feat_type", default="FACE_REID_MNV2", dest="face_feat_type",
                        choices=["FACE_REID_MNV2", "MOBILE_FACENET_ONNX",
                                 "FACENET_OV", "FACENET_TRT", "FACENET_AGE_GENDER"],
                        help="Type of face feature extracter to use for tracking. (default: %(default)s)")
    parser.add_argument("--is", "--input_size", dest="input_size",
                        nargs=2, default=(400, 500),
                        help='Input images are resized to this (width, height) -is 640 640. (default: %(default)s).')
    parser.add_argument('--noface', '--dont_save_face', dest="dont_save_face",
                        action="store_false",
                        help="""Flag avoids saving faces if set.""")
    parser.add_argument('--nofeat', '--dont_save_feat', dest="dont_save_feat",
                        action="store_false",
                        help="""Flag avoids saving face feats if set.""")
    parser.add_argument("-p", "--prototxt", dest="prototxt",
                        default="weights/face_detection_caffe/deploy.prototxt.txt",
                        help="Path to 'deploy' prototxt file. (default: %(default)s)")
    args = parser.parse_args()
    logging.info(f"Arguments used: {args}")
    print("Current Arguments: ", args)
    net = load_net(model=args.model,
                   prototxt=args.prototxt,
                   feat_net_type=args.face_feat_type,
                   det_thres=args.det_thres,
                   bbox_area_thres=args.bbox_area_thres,
                   model_in_size=args.input_size,
                   device=args.device)
    filter_faces_from_data(args.source_datadir_path,
                           args.target_datadir_path,
                           net,
                           save_face=args.dont_save_face,
                           save_feat=args.dont_save_feat)


if __name__ == "__main__":
    main()
