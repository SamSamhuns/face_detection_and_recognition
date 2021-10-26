# script must be evoked from dir face_detection_and_extraction
import os
import sys
import pickle
import logging
from glob import glob
from pathlib import Path
from datetime import datetime
from collections import Counter

import cv2
import onnxruntime
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat

sys.path.append(".")
from modules.common_utils import get_argparse, check_img_size
from modules.common_utils import read_pickle, write_json
from modules.yolov5_face.onnx.onnx_utils import inference_onnx_model_yolov5_face, get_bboxes_and_confs
from modules.mobile_facenet.utils import inference_onnx_model_mobile_facenet

today = datetime.today()
year, month, day, hour, minute, sec = today.year, today.month, today.day, today.hour, today.minute, today.second
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename=f'logs/imdb_wiki_face_extract_statistics_{year}{month}{day}_{hour}:{minute}:{sec}.log',
                    level=logging.INFO)


class Net(object):
    __slots__ = ["face_net", "feat_net", "det_thres",
                 "bbox_area_thres", "face_det_in_size", "face_feat_in_size"]

    def __init__(self, face_net, feat_net, det_thres, bbox_area_thres, face_det_in_size, face_feat_in_size):
        self.face_net = face_net
        self.feat_net = feat_net
        self.det_thres = det_thres
        self.bbox_area_thres = bbox_area_thres
        # in_size = (width, height), conv to int
        face_det_in_size = tuple(map(int, face_det_in_size))
        # input size must be multiple of max stride 32 for yolov5 models
        self.face_det_in_size = tuple(map(check_img_size, face_det_in_size))
        self.face_feat_in_size = tuple(map(int, face_feat_in_size))


def calc_age(taken, dob):
    """Copied from
    https://github.com/yu4u/age-gender-estimation/blob/master/src/utils.py
    """
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def get_metadata_from_imdb_wiki(mat_path, db):
    """Copied from
    https://github.com/yu4u/age-gender-estimation/blob/master/src/utils.py
    """
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    assert len(full_path) == len(dob) == len(gender) == len(
        photo_taken) == len(face_score) == len(second_face_score) == len(age)
    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def load_net(face_det_onnx_path, face_feat_onnx_path, det_thres, bbox_area_thres, face_det_in_size, face_feat_in_size):
    # load face det net
    face_det_net = onnxruntime.InferenceSession(face_det_onnx_path)
    # load face feat ext net
    feat_ext_net = onnxruntime.InferenceSession(face_feat_onnx_path)
    return Net(face_det_net, feat_ext_net, det_thres, bbox_area_thres, face_det_in_size, face_feat_in_size)


def extract_imdb_wiki_yolov5(dataset_path, net):
    dataset = dataset_path.split('/')[-1]
    if dataset not in {'imdb', 'wiki'}:
        raise NotImplementedError(f"Extraction for {dataset} not supported")

    logging.info("Extracting faces and feature vectors ")
    image_paths = glob(f'{dataset_path}/*/*.jpg')
    fail_count = 0
    for image_path in tqdm(image_paths):
        try:
            image = cv2.imread(image_path)
        except Exception as e:
            logging.error(f"Failed to read image from {image_path}: {e}")
            fail_count += 1
            continue
        try:
            detections = inference_onnx_model_yolov5_face(
                net.face_net, image, net.face_det_in_size)
        except Exception as e:
            logging.error(
                f"Failed to run yolov5-face inference on {image_path}: {e}")
            fail_count += 1
            continue
        if detections is None:
            logging.error(f"Failed to detect faces for {image_path}")
            fail_count += 1
            continue

        iw, ih = net.face_det_in_size
        h, w = image.shape[:2]
        boxes, confs = get_bboxes_and_confs(
            detections, net.det_thres, net.bbox_area_thres, orig_size=(w, h), in_size=(iw, ih))
        face_feats = []
        tx, ty = -10, -1
        bx, by = 10, 5
        for box in boxes:
            xmin, ymin, xmax, ymax = box.astype('int')
            x, y, xw, yh = xmin + tx, ymin + ty, xmax + bx, ymax + by
            x, y, xw, yh = max(x, 0), max(y, 0), min(xw, w), min(yh, h)
            face = image[y:yh, x:xw].copy()
            face = cv2.resize(face, (net.face_feat_in_size))
            face_feat = inference_onnx_model_mobile_facenet(
                net.feat_net, face, net.face_feat_in_size)
            normed_feat = face_feat / np.linalg.norm(face_feat)
            face_feats.append(normed_feat)

        with open(image_path + '.pkl', 'wb') as stream:
            faces = []
            for conf, feat in zip(confs, face_feats):
                face_info_dict = {}
                face_info_dict["det_score"] = conf
                face_info_dict["normed_feature"] = feat
                faces.append(face_info_dict)
            pickle.dump(faces, stream)

    logging.error(
        f"in total {fail_count} number of images failed to extract face features out of {len(image_paths)} images")
    logging.info("Face and feature vector extraction complete.")


def clean_imdb_wiki(dataset_path: str, det_score: float = 0.8):
    """Copied from
    https://github.com/yu4u/age-gender-estimation/blob/master/create_db.py
    """
    dataset = dataset_path.split('/')[-1]
    if dataset not in {'imdb', 'wiki'}:
        raise NotImplementedError(f"Data cleaning for {dataset} not supported")

    logging.debug(f"Getting clean data from {dataset_path} ...")
    root_dir = Path('./')
    data_dir = root_dir.joinpath(dataset_path)
    mat_path = data_dir.joinpath(f"{dataset}.mat")

    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_metadata_from_imdb_wiki(
        mat_path, dataset)

    genders = []
    ages = []
    img_paths = []
    face_info_pkl_paths = []
    sample_num = len(face_score)

    metadata = {'total_num_images': len(full_path)}
    metadata['removed'] = {}
    metadata['removed']['age_not_correct'] = 0
    metadata['removed']['gender_not_correct'] = 0
    metadata['removed']['image_not_correct'] = 0
    metadata['removed']['no_face_detected'] = 0
    metadata['removed']['more_than_one_face'] = 0
    metadata['removed']['bad_quality'] = 0
    metadata['removed']['no_features'] = 0

    logging.debug(f"Extracting metadata from {dataset_path} ...")
    for i in tqdm(range(sample_num)):
        if ~(0 <= age[i] <= 100):
            metadata['removed']['age_not_correct'] += 1
            continue
        if np.isnan(gender[i]):
            metadata['removed']['gender_not_correct'] += 1
            continue

        img_path = str(data_dir / full_path[i][0])
        face_info_pkl_path = img_path + '.pkl'
        if not os.path.isfile(face_info_pkl_path):
            metadata['removed']['image_not_correct'] += 1
            continue
        fde_data = read_pickle(face_info_pkl_path)
        if fde_data is None:
            metadata['removed']['no_features'] += 1
            continue
        if len(fde_data) == 0:
            metadata['removed']['no_face_detected'] += 1
            continue
        if len(fde_data) > 1:
            metadata['removed']['more_than_one_face'] += 1
            continue
        if fde_data[0]['det_score'] < det_score:
            metadata['removed']['bad_quality'] += 1
            continue

        genders.append({0: 'f', 1: 'm'}[int(gender[i])])
        ages.append(int(age[i]))
        img_paths.append(img_path)
        face_info_pkl_paths.append(face_info_pkl_path)
    assert len(genders) == len(ages) == len(img_paths) == len(face_info_pkl_paths)

    data = []
    logging.debug(f"Saving {dataset} features ...")
    for gender, age, img_path, fde_path in tqdm(zip(genders, ages, img_paths, face_info_pkl_paths)):
        fde_data = read_pickle(fde_path)
        assert len(fde_data) == 1
        data_sample = {'image_path': img_path,
                       'age': age,
                       'gender': gender,
                       'feature': fde_data[0]['normed_feature']}
        data.append(data_sample)

    metadata['genders'] = dict(Counter(genders))
    metadata['ages'] = dict(Counter(ages))

    logging.info(f"{dataset}\'s metadata: {metadata}")
    metadata_write_path = str(data_dir / "meta-data.json")
    write_json(metadata, metadata_write_path)
    logging.info(f"{dataset}\'s metadata written at : {metadata_write_path}")

    data_write_path = str(data_dir / "data.npy")
    np.save(data_write_path, data)
    logging.info(
        f"{dataset}\'s data (features) written at : {data_write_path}")

    logging.info(f"Data cleaning complete for {dataset_path}")


def main():
    parser = get_argparse(
        description="IMDB-WIKI dataset face and face-feature extraction and cleaning", conflict_handler='resolve')
    parser.remove_arguments(["model", "prototxt", "input_src"])
    parser.add_argument("-d", "--dataset_path",
                        default="data/wiki", choices=["data/imdb", "data/wiki"],
                        help="Dataset type. (default: %(default)s)")
    parser.add_argument("-md", "--model_det",
                        default="weights/yolov5s/yolov5s-face.onnx",
                        help='Path to yolov5-face detection onnx file. (default: %(default)s).')
    parser.add_argument("-mf", "--model_feat",
                        default="weights/mobile_facenet/mobile_facenet.onnx",
                        help='Path to face feature extracter onnx file. (default: %(default)s).')
    parser.add_argument("-id", "--face_det_in_size",
                        nargs=2,
                        default=(640, 640),
                        help='Input images are resized to this size (w, h) for face det. (default: %(default)s).')
    parser.add_argument("-if", "--face_feat_in_size",
                        nargs=2,
                        default=(112, 112),
                        help='Face images are resized to this size (w, h) for feature ext. (default: %(default)s).')
    args = parser.parse_args()
    print("Current Arguments: ", args)
    net = load_net(args.model_det, args.model_feat,
                   args.det_thres, args.bbox_area_thres,
                   args.face_det_in_size, args.face_feat_in_size)
    extract_imdb_wiki_yolov5(args.dataset_path, net)
    clean_imdb_wiki(args.dataset_path)


if __name__ == "__main__":
    main()
