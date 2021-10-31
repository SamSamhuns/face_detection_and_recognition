import numpy as np
import cv2
import os

from modules.common_utils import get_argparse, get_file_type, draw_bbox_on_image
from modules.openvino.utils import OVNetwork
from modules.opencv2_dnn.utils import get_bboxes_confs_areas


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

    h, w = image.shape[:2]
    ih, iw = net.in_shape[2:4]
    # pass the image through the network to obtain the detections
    detections = net.inference_img(image)[0][0]
    boxes, confs, areas = get_bboxes_confs_areas(
        detections, net.det_thres, net.bbox_area_thres, (w, h), (iw, ih))
    draw_bbox_on_image(image, boxes, confs, areas)

    cv2.imshow("openvino", image)
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
    parser = get_argparse(description="OpenVINO face detection")
    parser.remove_arguments(["model", "prototxt"])
    parser.add_argument("-mb", "--model_bin_path",
                        default="weights/face-detection-0204/face-detection-0204.bin",
                        help="Path to openVINO model BIN file. (default: %(default)s)")
    parser.add_argument("-mx", "--model_xml_path",
                        default="weights/face-detection-0204/face-detection-0204.xml",
                        help="Path to openVINO model XML file. (default: %(default)s)")
    args = parser.parse_args()

    net = OVNetwork(args.model_xml_path, args.model_bin_path,
                    args.det_thres, args.bbox_area_thres, device="CPU")
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
