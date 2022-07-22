from modules.utils.parser import get_argparse
from modules.utils.files import get_file_type
from modules.blazeface.model import BlazeFaceModel
from modules.models.inference import inference_img, inference_vid, inference_webcam


def main():
    parser = get_argparse(
        description="Blazeface face detection", conflict_handler='resolve')
    parser.add_argument("--md", "--model", dest="model",
                        default="weights/blazeface/blazefaceback.pth",
                        help=("Path to weight file (.pth/.onnx). (default: %(default)s). "
                              "anchors should be placed in the same dir as weights. "
                              "anchorsback.npy for model_type == 'back' else anchors.npy"))
    parser.add_argument("--mt", "--model_type", dest="model_type",
                        default="back", choices=["back", "front"],
                        help=("Model type back or front. The --md model weight file must also match. (default: %(default)s). "
                              "anchors should be placed in the same dir as weights."))
    args = parser.parse_args()
    net = BlazeFaceModel(
        args.model, args.det_thres, args.bbox_area_thres, args.model_type, args.device)

    # choose inference mode
    input_type = get_file_type(args.input_src)
    if input_type == "camera":
        inference_webcam(net, int(args.input_src), "blazeface")
    elif input_type == "video":
        inference_vid(net, args.input_src, "blazeface")
    elif input_type == "image":
        inference_img(net, args.input_src, "blazeface")
    else:
        print("File type or inference mode not recognized. Use --help")


if __name__ == "__main__":
    main()
