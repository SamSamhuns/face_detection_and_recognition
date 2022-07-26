from modules.openvino.model import OVModel
from modules.utils.parser import get_argparse
from modules.utils.files import get_file_type
from modules.utils.inference import inference_img, inference_vid, inference_webcam


def main():
    parser = get_argparse(description="OpenVINO face detection")
    parser.remove_arguments(["model"])
    parser.add_argument("--mb", "--model_bin_path", dest="model_bin_path",
                        default="weights/face_detection_0204/model.bin",
                        help="Path to openVINO model BIN file. (default: %(default)s)")
    parser.add_argument("--mx", "--model_xml_path", dest="model_xml_path",
                        default="weights/face_detection_0204/model.xml",
                        help="Path to openVINO model XML file. (default: %(default)s)")
    args = parser.parse_args()

    net = OVModel(args.model_xml_path, args.model_bin_path,
                  args.det_thres, args.bbox_area_thres, device=args.device.upper())
    # choose inference mode
    input_type = get_file_type(args.input_src)
    if input_type == "camera":
        inference_webcam(net, int(args.input_src), "OpenVINO")
    elif input_type == "video":
        inference_vid(net, args.input_src, "OpenVINO")
    elif input_type == "image":
        inference_img(net, args.input_src, "OpenVINO")
    else:
        print("File type or inference mode not recognized. Use --help")


if __name__ == "__main__":
    main()
