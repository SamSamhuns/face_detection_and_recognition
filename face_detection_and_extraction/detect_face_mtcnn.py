from modules.utils.parser import get_argparse
from modules.utils.files import get_file_type
from modules.mtcnn.model import MTCNNSlowModel, MTCNNFastModel
from modules.utils.inference import inference_img, inference_vid, inference_webcam


def load_model(model_type, det_thres, bbox_area_thres, device):
    if model_type == "fast":
        device = "cpu:0" if "cpu" in device.lower() else device
        net = MTCNNFastModel(
            "weights/tf_mtcnn_fast/mtcnn.pb", det_thres, bbox_area_thres, device=device)
    elif model_type == "slow":
        net = MTCNNSlowModel(
            det_thres, bbox_area_thres)
    else:
        raise NotImplementedError(f"{model_type} is not supported")
    return net


def main():
    parser = get_argparse(description="MTCNN face detection")
    parser.remove_arguments(["model"])
    parser.add_argument("--mt", "--model_type", dest="model_type",
                        default="fast", choices=["fast", "slow"],
                        help=("MTCNN model type, fast or slow (default: %(default)s)."))

    args = parser.parse_args()
    print("Current Arguments: ", args)

    # load model
    net = load_model(args.model_type, args.det_thres, args.bbox_area_thres, args.device)

    # choose inference mode
    input_type = get_file_type(args.input_src)
    if input_type == "camera":
        inference_webcam(net, int(args.input_src), "mtcnn")
    elif input_type == "video":
        inference_vid(net, args.input_src, "mtcnn")
    elif input_type == "image":
        inference_img(net, args.input_src, "mtcnn")
    else:
        print("File type or inference mode not recognized. Use --help")


if __name__ == "__main__":
    main()
