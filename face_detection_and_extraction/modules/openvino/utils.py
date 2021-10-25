from modules.common_utils import pad_resize_image
from openvino.inference_engine import IECore


class OVNetwork(object):

    __slots__ = ["OVExec", "in_layer", "out_layer",
                 "det_thres", "bbox_area_thres", "in_shape", "out_shape"]

    def __init__(self, xml_path, bin_path, det_thres, bbox_area_thres, device="CPU"):
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

        # print model input/output info and shapes
        print("Available Devices: ", OVIE.available_devices)
        print("Input Layer: ", self.in_layer)
        print("Output Layer: ", self.out_layer)
        print("Input Shape: ", self.in_shape)
        print("Output Shape: ", self.out_shape)

    def inference_img(self, cv2_img, preprocess_func=pad_resize_image):
        # preprocess
        N, C, H, W = self.in_shape
        resized = preprocess_func(cv2_img, (W, H))  # padded resize
        resized = resized.transpose((2, 0, 1))  # HWC to CHW
        in_image = resized.reshape((N, C, H, W))
        # openVINO expects BGR format
        detections = self.OVExec.infer(inputs={self.in_layer: in_image})
        return detections[self.out_layer]
