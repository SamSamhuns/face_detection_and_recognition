from modules.utils.parser import get_argparse
from modules.utils.files import get_file_type
from modules.mtcnn.model import MTCNNModel
from modules.utils.inference import inference_img, inference_vid, inference_webcam


def main():
    parser = get_argparse(description="MTCNN face detection")
    parser.remove_arguments(["model", "device"])
    args = parser.parse_args()

    # load model
    net = MTCNNModel(args.det_thres, args.bbox_area_thres)

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
