#  splits a directory with object classes in different subdirectories into
#  train, test and optionally val sub-directory with the same class sub-dir
#  structure
import os
import cv2
import glob
import logging
import mimetypes
import onnxruntime
import numpy as np
from tqdm import tqdm
from datetime import datetime

from modules.common_utils import get_argparse, fix_path_for_globbing, check_img_size
from modules.opencv2_dnn.utils import inference_cv2_model as inf_cv2
from modules.opencv2_dnn.utils import get_bboxes_and_confs as get_bboxes_confs_cv2
from modules.yolov5_face.onnx.onnx_utils import inference_onnx_model_yolov5_face as inf_yolov5
from modules.yolov5_face.onnx.onnx_utils import get_bboxes_and_confs as get_bboxes_confs_yolov5


mimetypes.init()
today = datetime.today()
year, month, day, hour, minute, sec = today.year, today.month, today.day, today.hour, today.minute, today.second

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename=f'logs/extraction_statistics_{year}{month}{day}_{hour}:{minute}:{sec}.log',
                    level=logging.INFO)

# ######################## Settings ##################################

SAVE_VIDEO_FACES_IN_SUBDIRS = True
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
    __slots__ = ["face_net", "inf_func", "bbox_conf_func",
                 "det_thres", "bbox_area_thres",
                 "FACE_MODEL_MEAN_VALUES", "FACE_MODEL_INPUT_SIZE", "FACE_MODEL_OUTPUT_SIZE"]

    def __init__(self, face_net, inf_func, bbox_conf_func, det_thres, bbox_area_thres,
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


def get_img_vid_media_type(fname):
    mimestart = mimetypes.guess_type(fname)[0]
    if mimestart is not None:
        mimestart = mimestart.split('/')[0]
        if mimestart in ['video', 'image']:
            return mimestart
    return None


def load_net(model, prototxt, det_thres, bbox_area_thres, model_in_size, model_out_size, device="cpu"):
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
        bbox_conf_func = get_bboxes_confs_yolov5
    else:
        if device == "cpu":
            face_net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU device")
        elif device == "gpu":
            face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using GPU device")
        inf_func = inf_cv2
        bbox_conf_func = get_bboxes_confs_cv2
    return Net(face_net, inf_func, bbox_conf_func,
               det_thres, bbox_area_thres,
               model_in_size, model_out_size)


def extract_face_and_conf_list(net, img):
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
    boxes, confs = net.bbox_conf_func(
        detections, net.det_thres, net.bbox_area_thres, orig_size=(w, h), in_size=(iw, ih))

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
        if net.FACE_MODEL_OUTPUT_SIZE is not None:
            face = cv2.resize(face, (net.FACE_MODEL_OUTPUT_SIZE))
        face_list.append(face)
    return face_list, confs


def save_extracted_faces(media_prefix_name_list, faces_img_list, faces_conf_list, faces_save_dir) -> None:
    """
    args;
        face_img_list: list of cropped faces as np.ndarray
        target_dir: dir where face imgs are saved in class dirs
        class_name: name of class
    """
    total = 0
    for prefix, face_img_list, face_conf_list in zip(media_prefix_name_list, faces_img_list, faces_conf_list):
        i = 0
        for face_img, face_conf in zip(face_img_list, face_conf_list):
            i += 1
            face_conf = round(face_conf, 3)
            cv2.imwrite(
                f"{faces_save_dir}/{prefix}_face_{i}_conf_{str(face_conf).replace('.', '_')}.jpg", face_img)
        total += i
    return total


def filter_faces_from_data(raw_img_dir, target_dir, net):
    os.makedirs(target_dir, exist_ok=True)
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
            os.makedirs(faces_save_dir, exist_ok=True)

            faces_img_list = []
            faces_conf_list = []
            media_prefix_name_list = []
            media_root = os.path.basename(media_path).split('.')[0]
            mtype = get_img_vid_media_type(media_path)
            if mtype == "image":
                media_prefix_name_list.append(media_root)
                faces, confs = extract_face_and_conf_list(net, media_path)
                faces_img_list.append(faces)
                faces_conf_list.append(confs)
            elif mtype == "video":
                # save faces from videos inside sub dirs if flag is set
                if SAVE_VIDEO_FACES_IN_SUBDIRS:
                    faces_save_dir = os.path.join(faces_save_dir, media_root)
                    if os.path.exists(faces_save_dir):  # skip pre-extracted faces
                        print(
                            f"Skipping {faces_save_dir} as it already exists.")
                        continue
                    os.makedirs(faces_save_dir, exist_ok=True)

                cap = cv2.VideoCapture(media_path)
                step = int(round(cap.get(cv2.CAP_PROP_FPS)))
                i = 0
                save_frames_num = 0
                ret, frame = cap.read()
                while ret:
                    i += 1
                    if i % step == 0 or i == 1:
                        save_frames_num += 1
                        if save_frames_num > MAX_N_FRAME_FROM_VID:
                            break
                        mprefix = '' if SAVE_VIDEO_FACES_IN_SUBDIRS else media_root + '_'
                        media_prefix_name_list.append(
                            mprefix + f"sec_{i//step}_")
                        faces, confs = extract_face_and_conf_list(net, frame)
                        faces_img_list.append(faces)
                        faces_conf_list.append(confs)
                    ret, frame = cap.read()
                cap.release()
                cv2.destroyAllWindows()

            faces_extracted = save_extracted_faces(
                media_prefix_name_list, faces_img_list, faces_conf_list, faces_save_dir)
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
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "gpu"],
                        help="Device to inference on. (default: %(default)s)")
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
