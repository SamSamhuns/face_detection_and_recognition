from openvino.inference_engine import IECore
import numpy as np
import cv2
import os

from modules.common_utils import get_argparse


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
    resized = cv2.resize(cv2_img, (W, H))
    resized = resized.transpose((2, 0, 1))  # HWC to CHW
    input_image = resized.reshape((N, C, H, W))
    detections = net.OVExec.infer(inputs={net.input_layer: input_image})
    return detections


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

    fh, fw = image.shape[:2]
    # pass the image through the bnetwork obtain the detections
    detections = inference_model(net, image)
    count = 0
    detections = detections[net.output_layer][0][0]
    for detection in detections:
        if detection[2] > threshold:
            count += 1
            xmin = int(detection[3] * fw)
            ymin = int(detection[4] * fh)
            xmax = int(detection[5] * fw)
            ymax = int(detection[6] * fh)
            cv2.rectangle(image, (xmin, ymin),
                          (xmax, ymax), (0, 125, 255), 3)
            text = f'label:{int(detection[1])}, conf:{round(detection[2], 2)}'
            cv2.putText(image, text, (xmin, ymin - 7),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 125, 255), 1)
    # print(f"Num of faces detected = {count} ")
    # show the output image
    cv2.imshow("output", image)
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

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def inference_webcam(net, threshold):
    inference_vid(net, 0, threshold)


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
    if args.webcam and args.image is None and args.video is None:
        inference_webcam(net, args.threshold)
    elif args.image and args.video is None and args.webcam is False:
        inference_img(net, args.image, args.threshold)
    elif args.video and args.image is None and args.webcam is False:
        inference_vid(net, args.video, args.threshold)
    else:
        print("Only one mode is allowed")
        print("\tpython detect_face_openvino.py -w              # webcam mode")
        print("\tpython detect_face_openvino.py -i img_path     # image mode")
        print("\tpython detect_face_openvino.py -v vid_path     # video mode")


if __name__ == "__main__":
    main()
