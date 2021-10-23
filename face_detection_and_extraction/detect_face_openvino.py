import numpy as np
import cv2
import os

from modules.common_utils import get_argparse, get_file_type
from modules.common_utils import scale_coords, draw_bbox_on_image
from modules.openvino.utils import OVNetwork


def inference_img(net, img, threshold, waitKey_val=0):
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
    # pass the image through the bnetwork obtain the detections
    detections = net.inference_img(image)[0][0]

    # filter dtections below threshold
    detections = detections[detections[:, 2] > threshold]
    # rescale detections to orig image size taking the padding into account
    N, C, H, W = net.in_shape
    boxes = detections[:, 3:7] * np.array([W, H, W, H])
    boxes = scale_coords((H, W), boxes, (h, w)).round()
    confs = detections[:, 2]
    draw_bbox_on_image(image, boxes, confs)

    cv2.imshow("openvino", image)
    cv2.waitKey(waitKey_val)


def inference_vid(net, vid, threshold):
    cap = cv2.VideoCapture(vid)
    ret, frame = cap.read()

    while ret:
        # inference and display the resulting frame
        inference_img(net, frame, threshold, waitKey_val=5)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()


def inference_webcam(net, cam_index, threshold):
    inference_vid(net, cam_index, threshold)


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

    net = OVNetwork(args.model_xml_path, args.model_bin_path, device="CPU")
    # choose inference mode
    input_type = get_file_type(args.input_src)
    if input_type == "camera":
        inference_webcam(net, int(args.input_src), args.threshold)
    elif input_type == "video":
        inference_vid(net, args.input_src, args.threshold)
    elif input_type == "image":
        inference_img(net, args.input_src, args.threshold)
    else:
        print("File type or inference mode not recognized. Use --help")


if __name__ == "__main__":
    main()
