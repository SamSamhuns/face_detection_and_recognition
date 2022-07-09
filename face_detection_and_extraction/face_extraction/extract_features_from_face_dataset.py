import os
import cv2
import sys
import glob
import time
import logging
import argparse
import mimetypes
import onnxruntime
import numpy as np
import os.path as osp
from tqdm import tqdm
from datetime import datetime

sys.path.append(".")
from modules.files import get_file_type, read_pickle
from modules.facenet_trt_server.inference import TritonServerInferenceSession as face_feat_trt_sess
from modules.openvino.utils import OVNetwork


mimetypes.init()
today = datetime.today()
year, month, day, hour, minute, sec = today.year, today.month, today.day, today.hour, today.minute, today.second

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename=f'logs/face_feat_ext_stats_{year}{month}{day}_{hour}:{minute}:{sec}.log',
                    level=logging.INFO)

# ######################## Settings ##################################

CLASS_NAME_TO_LABEL_DICT = read_pickle(
    "data/class_name_to_label.pkl")
CLASSES_TO_EXCLUDE = {}

# size of features from one face
FACE_FEATURE_SIZE = 256
# max number of faces to consider from each frame for feat ext
MAX_N_FACES_PER_FRAME = 5
# max number of frames from which faces are extracted
MAX_N_FRAME_FROM_VID = 20
VALID_FILE_EXTS = {'jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm',
                   'mp4', 'avi'}


# #################### Raw Data Organization #########################
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
#
# example raw data path    = "raw_data/dataset"
# example target data path = "target_data/dataset"
# ###################################################################


class Net(object):
    __slots__ = ["feat_net_type", "feature_net"]

    def __init__(self, feat_net_type):
        self.feat_net_type = feat_net_type
        if feat_net_type == "MOBILE_FACENET_ONNX":
            self.feature_net = onnxruntime.InferenceSession(
                "weights/MOBILE_FACENET/MOBILE_FACENET.onnx")
        elif feat_net_type == "FACE_REID_MNV2":
            self.feature_net = OVNetwork(
                xml_path="weights/face_reidentification_retail_0095/FP32/model.xml",
                bin_path="weights/face_reidentification_retail_0095/FP32/model.bin",
                det_thres=None, bbox_area_thres=None, verbose=False)
        elif feat_net_type == "FACENET_OV":
            self.feature_net = OVNetwork(
                xml_path="weights/facenet_20180408_102900/facenet_openvino/20180408-102900.xml",
                bin_path="weights/facenet_20180408_102900/facenet_openvino/20180408-102900.bin",
                det_thres=None, bbox_area_thres=None, verbose=False)
        elif feat_net_type == "FACENET_TRT":
            self.feature_net = face_feat_trt_sess()
        else:
            raise NotImplementedError(
                f"""{feat_net_type} feature extraction net is not implemented""")

    def get_face_features(self, face):
        if self.feat_net_type == "MOBILE_FACENET_ONNX":
            feats = self._get_face_features_mobile_facenet_onnx(face)
        elif self.feat_net_type == "FACE_REID_MNV2":
            feats = self._get_face_features_openvino(face)
        elif self.feat_net_type == "FACENET_OV":
            feats = self._get_face_features_openvino(face)
        elif self.feat_net_type == "FACENET_TRT":
            feats = self._get_face_features_trt(face)
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
        features = self.feature_net.inference_img(
            face, preprocess_func=cv2.resize).astype(np.float32)[0]
        return features.squeeze()


def save_extracted_feat(feat, media_root, class_name, feats_save_dir):
    """
    args;
        feat: face feat numpy arr
        media_root: root name of media file
        class_name: name of class
        feats_save_dir: dir where face feats are saved in class dirs
    """
    annot_dict = {"media_id": media_root}
    os.makedirs(feats_save_dir, exist_ok=True)
    npy_savepath = os.path.join(feats_save_dir, media_root + ".npy")
    annot_dict["class_name"] = class_name
    annot_dict["label"] = CLASS_NAME_TO_LABEL_DICT[class_name]
    annot_dict["feature"] = feat
    np.save(npy_savepath, annot_dict)


def extract_features_from_face_data(source_dir, target_dir, net):
    tm0 = time.time()
    dir_list = glob.glob(osp.join(source_dir, "*"))

    total_feats_ext = 0
    # for each class in raw data
    for dir_path in tqdm(dir_list):  # iter through class dirs
        if not osp.isdir(dir_path):  # skip if path is not a dir
            continue
        class_name = dir_path.split("/")[-1]  # get class name
        if class_name in CLASSES_TO_EXCLUDE:
            print(f"Excluding extraction from class {class_name}")
            continue

        print(f"Features will now be extracted from class {class_name}")
        file_path_list = [file for file in glob.glob(osp.join(dir_path, "*"))
                          if file.split(".")[-1] in VALID_FILE_EXTS]

        class_feats_ext = 0
        # foreach image in file_path_list
        for media_path in tqdm(file_path_list):
            try:
                # create dir for saving feats per class
                feats_save_dir = osp.join(
                    target_dir, f"npy_feat_{FACE_FEATURE_SIZE}", class_name)

                media_root = osp.basename(media_path).split('.')[0]
                mtype = get_file_type(media_path)
                if mtype == "image":
                    feat = net.get_face_features(cv2.imread(media_path))
                    save_extracted_feat(
                        feat, media_root, class_name, feats_save_dir)
                    class_feats_ext += 1
            except Exception as e:
                print(f"{e}. Extraction failed for media {media_path}")
        total_feats_ext += class_feats_ext
        logging.info(f"{class_feats_ext} feats found for class {class_name}")
    logging.info(f"{total_feats_ext} feats extracted")
    logging.info(f"Total time taken: {time.time() - tm0:.2f}s")
    print(
        f"Time for getting {total_feats_ext} feats is {time.time() - tm0:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Dataset face extraction")
    parser.add_argument('-sd', '--source_datadir_path',
                        type=str,
                        default="data/face_data",
                        help="""Source dataset dir path with
                        class imgs inside folders.""")
    parser.add_argument('-td', '--target_datadir_path',
                        type=str, default="data/feat_256_train_age/train",
                        help="""Target dataset dir path where
                        face feats will be saved to. (default: %(default)s)""")
    parser.add_argument("-ft", "--face_feat_type", default="FACE_REID_MNV2",
                        choices=["FACE_REID_MNV2", "MOBILE_FACENET_ONNX",
                                 "FACENET_OV", "FACENET_TRT", "FACENET_AGE_GENDER"],
                        help="Face feature extracter type. (default: %(default)s)")
    args = parser.parse_args()
    logging.info(f"Arguments used: {args}")
    print("Current Arguments: ", args)
    net = Net(args.face_feat_type)
    extract_features_from_face_data(
        args.source_datadir_path, args.target_datadir_path, net)


if __name__ == "__main__":
    main()
