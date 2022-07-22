import os

import torch
import numpy as np

from modules.models.base import Model
from modules.blazeface.blazeface import BlazeFace
from modules.utils.image import pad_resize_image
from modules.blazeface.onnx_export import preprocess_onnx


MODEL_IN_SIZES = {"back": (256, 256),
                  "front": (128, 128)}


def load_net(model_path: str, model_type: str, device: str):
    print(f"Using {model_type} type model")
    # load face detection model
    _, fext = os.path.splitext(model_path)
    # anchors should be in the same dir as the blazeface pth weights
    anchors = "anchors.npy" if True else "anchorsback.npy"
    anchors = os.path.join(os.path.dirname(model_path), anchors)
    is_back_model = True if model_type == "back" else False
    device = "cpu" if not torch.cuda.is_available() else device
    if fext == ".pth":
        net = BlazeFace(back_model=is_back_model).to(device)
        net.load_weights(model_path)
        net.load_anchors(anchors)
        runtime = None
    elif fext == ".onnx":
        import onnxruntime
        net = BlazeFace(back_model=is_back_model)
        net.load_anchors(anchors, use_numpy=True)
        runtime = onnxruntime.InferenceSession(model_path, None)
    else:
        raise NotImplementedError(
            f"[ERROR] model with extension {fext} not implemented")
    return net, runtime


class BlazeFaceModel(Model):

    __slots__ = ["net", "runtime", "input_size",
                 "det_thres", "bbox_area_thres", "model_type"]

    def __init__(
            self,
            model_path: str,
            det_thres: float,
            bbox_area_thres: float,
            model_type: str,
            device: str = "cpu"):
        # input sizes are fixed based on model type
        input_size = MODEL_IN_SIZES[model_type]
        Model.__init__(self, input_size, det_thres, bbox_area_thres)

        self.net, self.runtime = load_net(model_path, model_type, device)
        self.model_type = model_type

    def __call__(self, cv2_img: np.ndarray):
        # preprocess
        resized = pad_resize_image(cv2_img, (self.input_size))  # padded resize

        if self.runtime is None:  # pytorch inference
            detections = self.inference_pytorch(resized)
        else:                     # onnx inference
            detections = self.inference_onnx(resized)
        # reorder dets to have [xmin, ymin, xmax, ymax, conf, landmarks...] fmt
        # from a [ymin, xmin, ymax, xmax, landmarks..., conf] fmt
        detections = detections[:, [1, 0, 3, 2, 16,
                                    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
        return detections

    def inference_pytorch(self, cv2_img: np.ndarray):
        img = cv2_img[..., ::-1]  # BGR to RGB
        detections = self.net.predict_on_image(img)
        detections = detections.cpu().numpy() if detections.cuda else detections
        return detections

    def inference_onnx(self, cv2_img: np.ndarray):
        img = preprocess_onnx(
            cv2_img, back_model=True if self.model_type == "back" else False)
        outputs = self.runtime.run(None, {"images": img})
        use_numpy = True

        # Postprocess the raw predictions:
        detections = self.net._tensors_to_detections(
            outputs[0], outputs[1], self.net.anchors, use_numpy=use_numpy)

        # Non-maximum suppression to remove overlapping detections:
        filtered_detections = []
        for detection in detections:
            faces = self.net._weighted_non_max_suppression(
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
