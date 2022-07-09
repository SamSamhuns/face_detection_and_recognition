import numpy as np
import torch
import cv2
import os

from modules.utils.parser import get_argparse
from modules.utils.files import get_file_type
from modules.utils.image import pad_resize_image
from modules.blazeface.onnx_export import preprocess_onnx
from modules.blazeface.blazeface import BlazeFace, plot_detections


def load_net(model):
    # load face detection model
    fpath, fext = os.path.splitext(model)
    back_model = True if fpath[-4:] == "back" else False
    anchors = "anchors.npy" if True else "anchorsback.npy"
    anchors = os.path.join(os.path.dirname(model), anchors)
    print(f"Using {'back' if back_model else 'front'} model")
    if fext == ".pth":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = BlazeFace(back_model=back_model).to(device)
        net.load_weights(model)
        net.load_anchors(anchors)
        runtime = None
    elif fext == ".onnx":
        import onnxruntime
        net = BlazeFace(back_model=back_model)
        net.load_anchors(anchors, use_numpy=True)
        runtime = onnxruntime.InferenceSession(model, None)
    else:
        raise NotImplementedError(
            f"[ERROR] model with extension {fext} not implemented")
    return net, runtime, back_model


def inference_pytorch_model(net, cv2_img, back_model=True):
    new_size = (256, 256) if back_model else (128, 128)
    cv2_img = cv2_img[..., ::-1]  # BGR to RGB
    img = pad_resize_image(cv2_img, new_size)
    detections = net.predict_on_image(img)
    detections = detections.cpu().numpy() if detections.cuda else detections
    return detections


def inference_onnx_model(net, runtime, cv2_img, back_model=True):
    new_size = (256, 256) if back_model else (128, 128)
    img = pad_resize_image(cv2_img, new_size)
    cv2.imshow('in', img)
    img = preprocess_onnx(img, back_model=back_model)
    outputs = runtime.run(None, {"images": img})
    use_numpy = True

    # Postprocess the raw predictions:
    detections = net._tensors_to_detections(
        outputs[0], outputs[1], net.anchors, use_numpy=use_numpy)

    # Non-maximum suppression to remove overlapping detections:
    filtered_detections = []
    for detection in detections:
        faces = net._weighted_non_max_suppression(
            detection, use_numpy=use_numpy)
        if use_numpy:
            faces = np.stack(faces) if len(faces) > 0 \
                else np.zeros((0, 17))
        else:
            faces = torch.stack(faces) if len(faces) > 0 \
                else torch.zeros((0, 17))
        filtered_detections.append(faces)

    detections = filtered_detections[0]
    return detections


def inference_img(net, runtime, back_model, img, threshold, waitKey_val=0):
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

    # pass the image through the network and
    # obtain the detections and predictions
    if runtime is None:  # pytorch inference
        detections = inference_pytorch_model(net, image, back_model)
    else:                # onnx inference
        detections = inference_onnx_model(net, runtime, image, back_model)

    model_in_HW = (256, 256) if back_model else (128, 128)
    plot_detections(
        image, detections, model_in_HW=model_in_HW, threshold=threshold, with_keypoints=True)

    # show the output image
    cv2.imshow("output", image)
    cv2.waitKey(waitKey_val)


def inference_vid(net, runtime, back_model, vid, threshold):
    cap = cv2.VideoCapture(vid)
    ret, frame = cap.read()

    while ret:
        inference_img(net, runtime, back_model,
                      frame, threshold, waitKey_val=5)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()


def inference_webcam(net, runtime, back_model, cam_index, threshold):
    inference_vid(net, runtime, back_model, cam_index, threshold)


def main():
    parser = get_argparse(
        description="Blazeface face detection", conflict_handler='resolve')
    parser.remove_arguments(["bbox_area_thres"])
    parser.add_argument("--md", "--model", dest="model",
                        default="weights/blazeface/blazefaceback.pth",
                        help=('Path to weight file (.pth/.onnx). (default: %(default)s). '
                              'anchors should be placed in the same dir as weights. '
                              'anchorsback.npy for back_model == True else anchors.npy'))
    args = parser.parse_args()

    net, runtime, back_model = load_net(args.model)
    # choose inference mode
    input_type = get_file_type(args.input_src)
    if input_type == "camera":
        inference_webcam(net, runtime, back_model,
                         int(args.input_src), args.det_thres)
    elif input_type == "video":
        inference_vid(net, runtime, back_model, args.input_src, args.det_thres)
    elif input_type == "image":
        inference_img(net, runtime, back_model, args.input_src, args.det_thres)
    else:
        print("File type or inference mode not recognized. Use --help")


if __name__ == "__main__":
    main()
