import numpy as np
import openvino as ov

from modules.models.base import Model
from modules.utils.image import pad_resize_image


class OVModel(Model):

    __slots__ = ["compiled_model", "out_layer_ir", "in_shape", "out_shape"]

    def __init__(
            self,
            xml_path: str,
            bin_path: str,
            det_thres: float,
            bbox_area_thres: float,
            device: str = "CPU",
            verbose: bool = False):
        core = ov.Core()
        model = core.read_model(model=xml_path, weights=bin_path)
        self.compiled_model = core.compile_model(model=model, device_name=device)

        self.det_thres = det_thres
        self.bbox_area_thres = bbox_area_thres
        # get input/output layer information
        self.out_layer_ir = self.compiled_model.output(model.outputs[0].names.pop())
        self.in_shape = model.inputs[0].shape
        self.out_shape = model.outputs[0].shape
        Model.__init__(self, self.in_shape[2:][::-1],
                       det_thres, bbox_area_thres)

        if verbose:
            # print model input/output info and shapes
            print("Available Devices: ", core.available_devices)
            print("Input Layer: ", model.inputs)
            print("Output Layer: ", model.outputs)
            print("Input Shape: ", self.in_shape)
            print("Output Shape: ", self.out_shape)

    def __call__(
            self,
            cv2_img: np.ndarray) -> np.ndarray:
        # preprocess
        N, C, H, W = self.in_shape
        resized = pad_resize_image(cv2_img, (W, H))  # padded resize
        resized = resized.transpose((2, 0, 1))  # HWC to CHW
        in_image = resized.reshape((N, C, H, W))
        # openVINO expects BGR format
        detections = self.compiled_model([in_image])
        detections = detections[self.out_layer_ir][0][0]
        # reorder dets to have [xmin, ymin, xmax, ymax, conf] format
        # from a [_, _, conf, xmin, ymin, xmax, ymax] fmt
        return detections[:, [3, 4, 5, 6, 2]]


class OVFeatModel(Model):

    __slots__ = ["compiled_model", "out_layer_ir", "in_shape", "out_shape"]

    def __init__(
            self,
            xml_path: str,
            bin_path: str,
            device: str = "CPU",
            verbose: bool = False):
        core = ov.Core()
        model = core.read_model(model=xml_path, weights=bin_path)
        self.compiled_model = core.compile_model(model=model, device_name=device)

        # get input/output layer information
        self.out_layer_ir = self.compiled_model.output(model.outputs[0].names.pop())
        self.in_shape =  model.inputs[0].shape
        self.out_shape =  model.outputs[0].shape
        Model.__init__(self, self.in_shape[2:][::-1],
                       det_thres=None, bbox_area_thres=None)

        if verbose:
            # print model input/output info and shapes
            print("Available Devices: ", core.available_devices)
            print("Input Layer: ", model.inputs)
            print("Output Layer: ", model.outputs)
            print("Input Shape: ", self.in_shape)
            print("Output Shape: ", self.out_shape)

    def __call__(
            self,
            cv2_img: np.ndarray) -> np.ndarray:
        # preprocess
        N, C, H, W = self.in_shape
        resized = pad_resize_image(cv2_img, (W, H))  # padded resize
        resized = resized.transpose((2, 0, 1))  # HWC to CHW
        in_image = resized.reshape((N, C, H, W))
        # openVINO expects BGR format
        detections = self.compiled_model([in_image])
        detections = detections[self.out_layer_ir].squeeze()
        return detections
