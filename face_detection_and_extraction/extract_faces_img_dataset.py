#  splits a directory with object classes in different subdirectories into
#  train, test and optionally val sub-directory with the same class sub-dir
#  structure
import os
import cv2
import glob
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime

from modules.utils import get_argparse, _fix_path_for_globbing

today = datetime.today()
year, month, day, hour, minute, sec = today.year, today.month, today.day, today.hour, today.minute, today.second

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename=f'logs/extraction_statistics_{year}_{month}_{day}_{hour}_{minute}_{sec}.log',
                    level=logging.INFO)

# #################### Raw Data Organization #########################
#   raw_data
#          |_ dataset
#                   |_ class_1
#                             |_ img1
#                             |_ img2
#                             |_ ....
#                   |_ class_2
#                             |_ img1
#                             |_ img2
#                             |_ ....
#                   ...
# ###################################################################

# #################### Data Configurations here #####################
# example raw data path = "data/raw_data/birds_dataset"
# example target data path = "data/processed_birds_dataset"
VALID_FILE_EXTS = {'jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm'}
# ###################################################################


def main():
    parser = get_argparse(description="Dataset face extraction")
    parser.remove_arguments(["image", "video", "webcam"])
    parser.add_argument('-rd', '--raw_data_path',
                        type=str, required=True,
                        help="""Raw dataset path with
                        class imgs inside folders. (default: %(default)s)""")
    parser.add_argument('-td', '--target_data_path',
                        type=str, required=True,
                        help="""Target dataset path where
                        imgs will be sep into train & test. (default: %(default)s)""")
    args = parser.parse_args()

    filter_faces_from_data(args.raw_data_path,
                           args.target_data_path,
                           args.model,
                           args.prototxt,
                           args.threshold)


def filter_faces_from_data(RAW_IMG_DIR, TARGET_IMG_DIR, MODEL, PROTO, THRES):
    os.makedirs(TARGET_IMG_DIR, exist_ok=True)
    dir_list = glob.glob(_fix_path_for_globbing(RAW_IMG_DIR))
    net = load_net(MODEL, PROTO)

    # for each class in raw data
    for i in tqdm(range(len(dir_list))):
        dir = dir_list[i]                # get path to class dir
        class_name = dir.split("/")[-1]  # get class name

        img_path_list = [file for file in glob.glob(dir + "/*")
                         if file.split(".")[-1] in VALID_FILE_EXTS]

        class_face_img_list = []
        for img_path in img_path_list:
            class_face_img_list.extend(extract_face_list(net, img_path, THRES))

        save_extracted_faces(class_face_img_list, TARGET_IMG_DIR, class_name)


def load_net(model, prototxt):
    # load face detection model
    fname, fext = os.path.splitext(model)
    if fext == ".caffemodel":
        net = cv2.dnn.readNetFromCaffe(prototxt, model)
    elif fext == ".pb":
        net = cv2.dnn.readNetFromTensorflow(model, prototxt)
    else:
        raise NotImplementedError(
            f"[ERROR] model with extension {fext} not implemented")
    return net


def extract_face_list(model, img_path, thres=0.5):
    """returns a list of cv2 images containing faces
    """
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(img_path, np.ndarray):
        img = img_path

    h, w = img.shape[:2]
    faces = inference_model(model, img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    tx, ty = -6, -1
    bx, by = 4, 5

    # copr faces from image
    face_list = []
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > thres:
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            # x, y  = top x left, top y left
            # xw, yh =  bottom x right, bottom y right
            # crop face, img[ty:by, tx:bx], img[y:yh, x:xw]
            x, y, xw, yh = x + tx, y + ty, x1 + bx, y1 + by
            x, y, xw, yh = max(x, 0), max(y, 0), min(xw, w), min(yh, h)
            # .copy() only keeps crops in memory
            face = img[y:yh, x:xw].copy()
            face_list.append(face)

    return face_list


def inference_model(net, cv2_img):
    blob = cv2.dnn.blobFromImage(cv2.resize(
        cv2_img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    return faces


def save_extracted_faces(face_img_list, target_dir, class_name):
    class_dir = os.path.join(target_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    i = 0
    for face_img in face_img_list:
        i += 1
        cv2.imwrite(f"{class_dir}/{i}.jpg", face_img)

    logging.info(f"{i} faces extracted for class {class_name}")


if __name__ == "__main__":
    main()
