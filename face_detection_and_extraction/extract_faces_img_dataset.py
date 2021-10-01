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

from modules.common_utils import get_argparse, fix_path_for_globbing
from modules.common_utils import pad_resize_image, scale_coords

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


class Net(object):
    __slots__ = ["face_net", "FACE_MODEL_MEAN_VALUES", "FACE_MODEL_INPUT_SIZE"]

    def __init__(self, face_net, model_in_size=(300, 300)):
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


def filter_faces_from_data(raw_img_dir, target_img_dir, net, threshold):
    os.makedirs(target_img_dir, exist_ok=True)
    dir_list = glob.glob(fix_path_for_globbing(raw_img_dir))

    # for each class in raw data
    for i in tqdm(range(len(dir_list))):
        dir = dir_list[i]                # get path to class dir
        class_name = dir.split("/")[-1]  # get class name

        img_path_list = [file for file in glob.glob(dir + "/*")
                         if file.split(".")[-1] in VALID_FILE_EXTS]

        class_face_img_list = []
        for img_path in img_path_list:
            class_face_img_list.extend(extract_face_list(net, img_path, threshold))

        save_extracted_faces(class_face_img_list, target_img_dir, class_name)


def extract_face_list(net, img_path, threshold=0.5):
    """returns a list of cv2 images containing faces
    """
    if isinstance(img_path, str):
        image = cv2.imread(img_path)
    elif isinstance(img_path, np.ndarray):
        image = img_path

    h, w = image.shape[:2]
    nw, nh = net.FACE_MODEL_INPUT_SIZE

    # pass the blob through the network and
    # obtain the detections and predictions
    detections = inference_model(net, image)[0][0]

    # filter dtections below threshold
    detections = detections[detections[:, 2] > threshold]
    # rescale detections to orig image size taking the padding into account
    boxes = detections[:, 3:7] * np.array([nw, nh, nw, nh])
    boxes = scale_coords((nh, nw), boxes, (h, w)).round()

    tx, ty = -6, -1
    bx, by = 4, 5

    # copy faces from image
    face_list = []
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box.astype('int')
        # x, y  = top x left, top y left
        # xw, yh =  bottom x right, bottom y right
        # crop face, image[ty:by, tx:bx], image[y:yh, x:xw]
        x, y, xw, yh = xmin + tx, ymin + ty, xmax + bx, ymax + by
        x, y, xw, yh = max(x, 0), max(y, 0), min(xw, w), min(yh, h)
        # .copy() only keeps crops in memory
        face = image[y:yh, x:xw].copy()
        face_list.append(face)
    return face_list


def inference_model(net, cv2_img):
    resized = pad_resize_image(cv2_img, net.FACE_MODEL_INPUT_SIZE)
    # opencv expects BGR format
    blob = cv2.dnn.blobFromImage(resized, 1.0,
                                 net.FACE_MODEL_INPUT_SIZE,
                                 net.FACE_MODEL_MEAN_VALUES)
    net.face_net.setInput(blob)
    faces = net.face_net.forward()
    return faces


def save_extracted_faces(face_img_list, target_dir, class_name):
    class_dir = os.path.join(target_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    i = 0
    for face_img in face_img_list:
        i += 1
        cv2.imwrite(f"{class_dir}/{i}.jpg", face_img)

    logging.info(f"{i} faces extracted for class {class_name}")


def main():
    parser = get_argparse(description="Dataset face extraction")
    parser.remove_argument("input_src")
    parser.add_argument('-rd', '--raw_data_path',
                        type=str, required=True,
                        help="""Raw dataset path with
                        class imgs inside folders. (default: %(default)s)""")
    parser.add_argument('-td', '--target_data_path',
                        type=str, required=True,
                        help="""Target dataset path where
                        imgs will be sep into train & test. (default: %(default)s)""")
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "gpu"],
                        help="Device to inference on. (default: %(default)s)")
    parser.add_argument("-is", "--input_size",
                        default=(300, 400),
                        help='Input images are resized to this (width, height). (default: %(default)s).')
    args = parser.parse_args()
    print("Current Arguments: ", args)

    net = load_net(args.model,
                   args.prototxt,
                   model_in_size=args.input_size,
                   device=args.device)

    filter_faces_from_data(args.raw_data_path,
                           args.target_data_path,
                           net,
                           args.threshold)


if __name__ == "__main__":
    main()
