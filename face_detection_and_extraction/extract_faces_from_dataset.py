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

from modules.common_utils import get_argparse, fix_path_for_globbing
from modules.common_utils import pad_resize_image, scale_coords
from modules.yolov5_face.onnx.onnx_utils import preprocess_image, conv_strides_to_anchors, w_non_max_suppression

mimetypes.init()
today = datetime.today()
year, month, day, hour, minute, sec = today.year, today.month, today.day, today.hour, today.minute, today.second

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename=f'logs/extraction_statistics_{year}{month}{day}_{hour}:{minute}:{sec}.log',
                    level=logging.INFO)

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
                 "FACE_MODEL_MEAN_VALUES", "FACE_MODEL_INPUT_SIZE", "FACE_MODEL_OUTPUT_SIZE"]

    def __init__(self, face_net, inf_func, bbox_conf_func, model_in_size=(640, 640), model_out_size=None):
        self.face_net = face_net
        self.inf_func = inf_func
        self.bbox_conf_func = bbox_conf_func
        # (width, height)
        self.FACE_MODEL_INPUT_SIZE = tuple(map(int, model_in_size))
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


def load_net(model, prototxt, model_in_size=(300, 300), model_out_size=(112, 112), device="cpu"):
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
        inf_func = inference_yolov5_onnx_model
        bbox_conf_func = get_bboxes_and_confs_from_yolov5_dets
    else:
        if device == "cpu":
            face_net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU device")
        elif device == "gpu":
            face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using GPU device")
        inf_func = inference_cv2_model
        bbox_conf_func = get_bboxes_and_confs_from_cv2_dets
    return Net(face_net, inf_func, bbox_conf_func, model_in_size, model_out_size)


def inference_cv2_model(net, cv2_img, input_size):
    resized = pad_resize_image(cv2_img, input_size=input_size)
    # opencv expects BGR format
    blob = cv2.dnn.blobFromImage(resized, 1.0,
                                 net.FACE_MODEL_INPUT_SIZE,
                                 net.FACE_MODEL_MEAN_VALUES)
    net.face_net.setInput(blob)
    detections = net.face_net.forward()
    return detections[0][0]


def inference_yolov5_onnx_model(net, cv2_img, input_size):
    resized = preprocess_image(cv2_img, input_size=input_size)
    outputs = net.face_net.run(None, {"images": resized})
    outputx = conv_strides_to_anchors(outputs, "cpu")
    detections = w_non_max_suppression(
        outputx, num_classes=1, conf_thres=0.4, nms_thres=0.3)
    return detections[0]


def get_bboxes_and_confs_from_cv2_dets(detections, threshold, orig_size, in_size):
    """
    Returns a tuple of bounding boxes and confidence scores
    """
    w, h = orig_size
    iw, ih = in_size
    # filter detections below threshold
    detections = detections[detections[:, 2] > threshold]
    confs = detections[:, 2]
    # rescale detections to orig image size taking the padding into account
    boxes = detections[:, 3:7] * np.array([iw, ih, iw, ih])
    boxes = scale_coords((ih, iw), boxes, (h, w)).round()

    return boxes, confs


def get_bboxes_and_confs_from_yolov5_dets(detections, threshold, orig_size, in_size):
    """
    Returns a tuple of bounding boxes and confidence scores
    """
    w, h = orig_size
    iw, ih = in_size
    # filter detections below threshold
    detections = detections[detections[..., 4] > threshold]
    confs = detections[..., 4].numpy()
    # rescale detections to orig image size taking the padding into account
    boxes = detections[..., :4].numpy()
    boxes = scale_coords((ih, iw), boxes, (h, w)).round()

    return boxes, confs


def extract_face_list(net, img, threshold=0.5):
    """returns a list of cv2 images containing faces
    """
    if isinstance(img, str):
        image = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        image = img

    h, w = image.shape[:2]
    iw, ih = net.FACE_MODEL_INPUT_SIZE

    # pass the blob through the network to get raw detections
    detections = net.inf_func(net, image, net.FACE_MODEL_INPUT_SIZE)
    if detections is None:  # no faces detected
        return []
    # obtain bounding boxesx and conf scores
    boxes, confs = net.bbox_conf_func(
        detections, threshold, orig_size=(w, h), in_size=(iw, ih))

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
    return face_list


def save_extracted_faces(media_prefix_name_list, faces_per_img_list, target_dir, class_name) -> None:
    """
    args;
        face_img_list: list of cropped faces as np.ndarray
        target_dir: dir where face imgs are saved in class dirs
        class_name: name of class
    """
    class_dir = os.path.join(target_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    total = 0

    for prefix, face_img_list in zip(media_prefix_name_list, faces_per_img_list):
        i = 0
        for face_img in face_img_list:
            i += 1
            cv2.imwrite(f"{class_dir}/{prefix}_face_{i}.jpg", face_img)
        total += i
    logging.info(f"{total} faces extracted for class {class_name}")


def filter_faces_from_data(raw_img_dir, target_dir, net, threshold):
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

        # foreach image or video in file_path_list
        for media_path in file_path_list:
            faces_img_list = []
            media_prefix_name_list = []
            mtype = get_img_vid_media_type(media_path)
            if mtype == "image":
                media_prefix_name_list.append(
                    os.path.basename(media_path).split('.')[0])
                faces_img_list.append(
                    extract_face_list(net, media_path, threshold))
            elif mtype == "video":
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
                        media_prefix_name_list.append(
                            os.path.basename(media_path).split('.')[0] + f"_sec_{i//step}_")
                        faces_img_list.append(
                            extract_face_list(net, frame, threshold))
                    ret, frame = cap.read()
                cap.release()
                cv2.destroyAllWindows()

            save_extracted_faces(
                media_prefix_name_list, faces_img_list, target_dir, class_name)


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

    net = load_net(args.model,
                   args.prototxt,
                   model_in_size=args.input_size,
                   model_out_size=args.output_size,
                   device=args.device)

    filter_faces_from_data(args.raw_datadir_path,
                           args.target_datadir_path,
                           net,
                           args.threshold)


if __name__ == "__main__":
    main()
