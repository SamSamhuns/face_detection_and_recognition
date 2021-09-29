from openvino.inference_engine import IECore
import numpy as np
import cv2
import os

from modules.common_utils import get_argparse, get_file_type
from modules.common_utils import pad_resize_image, scale_coords


class OVNetwork(object):

    __slots__ = ["OVExec", "OBJECT_DET_LABELS", "input_layer",
                 "output_layer", "input_shape", "output_shape"]

    def __init__(self, model_bin_path, model_xml_path, device="CPU"):
        OVIE = IECore()
        # load openVINO network
        OVNet = OVIE.read_network(
            model=model_xml_path, weights=model_bin_path)
        # create executable network
        self.OVExec = OVIE.load_network(
            network=OVNet, device_name=device)
        # dummy labels
        self.OBJECT_DET_LABELS = {i: "Object" for i in range(1000)}

        # get input/output layer information
        self.input_layer = next(iter(OVNet.input_info))
        self.output_layer = next(iter(OVNet.outputs))
        self.input_shape = OVNet.input_info[self.input_layer].input_data.shape
        self.output_shape = OVNet.outputs[self.output_layer].shape

        # print model input/output info and shapes
        print("Available Devices: ", OVIE.available_devices)
        print("Input Layer: ", self.input_layer)
        print("Output Layer: ", self.output_layer)
        print("Input Shape: ", self.input_shape)
        print("Output Shape: ", self.output_shape)


def inference_model(net, cv2_img):
    N, C, H, W = net.input_shape
    resized = pad_resize_image(cv2_img, (W, H))  # padded resize
    resized = resized.transpose((2, 0, 1))  # HWC to CHW
    input_image = resized.reshape((N, C, H, W))
    # openVINO expects BGR format
    detections = net.OVExec.infer(inputs={net.input_layer: input_image})
    return detections[net.output_layer]


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
    detections = inference_model(net, image)[0][0]

    # filter dtections below threshold
    detections = detections[detections[:, 2] > threshold]
    # rescale detections to orig image size taking the padding into account
    N, C, H, W = net.input_shape
    boxes = detections[:, 3:7] * np.array([W, H, W, H])
    boxes = scale_coords((H, W), boxes, (h, w)).round()

    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box.astype('int')
        cv2.rectangle(image, (xmin, ymin),
                      (xmax, ymax), (0, 125, 255), 3)
        text = f'label:{int(detections[i, 1])}, conf:{detections[i, 2]:.2f}'
        cv2.putText(image, text, (xmin, ymin - 7),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 125, 255), 1)
    # print(f"Num of faces detected = {i} ")
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

    net = OVNetwork(args.model_bin_path, args.model_xml_path, device="CPU")
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
