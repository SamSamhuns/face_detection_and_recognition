import numpy as np
from openvino.inference_engine import IECore

from modules.models.base import Model
from modules.utils.image import pad_resize_image


class OVModel(Model):

    __slots__ = ["OVExec", "in_layer", "out_layer",
                 "det_thres", "bbox_area_thres", "in_shape", "out_shape"]

    def __init__(
            self,
            xml_path: str,
            bin_path: str,
            det_thres: float,
            bbox_area_thres: float,
            device: str = "CPU",
            verbose: bool = False):
        OVIE = IECore()
        OVNet = OVIE.read_network(model=xml_path, weights=bin_path)

        self.OVExec = OVIE.load_network(network=OVNet, device_name=device)

        self.det_thres = det_thres
        self.bbox_area_thres = bbox_area_thres
        # get input/output layer information
        self.in_layer = next(iter(OVNet.input_info))
        self.out_layer = next(iter(OVNet.outputs))
        self.in_shape = OVNet.input_info[self.in_layer].input_data.shape
        self.out_shape = OVNet.outputs[self.out_layer].shape
        Model.__init__(self, self.in_shape[2:],
                       det_thres, bbox_area_thres)

        if verbose:
            # print model input/output info and shapes
            print("Available Devices: ", OVIE.available_devices)
            print("Input Layer: ", self.in_layer)
            print("Output Layer: ", self.out_layer)
            print("Input Shape: ", self.in_shape)
            print("Output Shape: ", self.out_shape)

    def __call__(
            self,
            cv2_img: np.ndarray):
        # preprocess
        N, C, H, W = self.in_shape
        resized = pad_resize_image(cv2_img, (W, H))  # padded resize
        resized = resized.transpose((2, 0, 1))  # HWC to CHW
        in_image = resized.reshape((N, C, H, W))
        # openVINO expects BGR format
        detections = self.OVExec.infer(inputs={self.in_layer: in_image})
        detections = detections[self.out_layer][0][0]
        # reorder dets to have [xmin, ymin, xmax, ymax, conf] format
        # from a [_, _, conf, xmin, ymin, xmax, ymax] fmt
        return detections[:, [3, 4, 5, 6, 2]]
