import cv2
import json
import numpy as np
from triton_python_backend_utils import get_input_tensor_by_name, Tensor
from triton_python_backend_utils import get_output_config_by_name, triton_string_to_numpy, InferenceResponse

from utils import conv_strides_to_anchors, w_non_max_suppression


class TritonPythonModel:
    def __init__(self) -> None:
        super().__init__()
        self.model_face_out_size = (112, 112)  # w,h

    def initialize(self, args):
        model_config = json.loads(args['model_config'])
        faces_config = get_output_config_by_name(
            model_config, "face_detector_faces")
        bboxs_config = get_output_config_by_name(
            model_config, "face_detector_bboxes")
        confs_config = get_output_config_by_name(
            model_config, "face_detector_confs")

        # Convert Triton types to numpy types
        self.face_dtype = triton_string_to_numpy(
            faces_config['data_type'])
        self.bbox_dtype = triton_string_to_numpy(
            bboxs_config['data_type'])
        self.conf_dtype = triton_string_to_numpy(
            confs_config['data_type'])

    def execute(self, requests):
        responses = []
        for request in requests:
            stride_8 = get_input_tensor_by_name(
                request, "stride_8_out").as_numpy()
            stride_16 = get_input_tensor_by_name(
                request, "stride_16_out").as_numpy()
            stride_32 = get_input_tensor_by_name(
                request, "stride_32_out").as_numpy()
            face_det_thres = get_input_tensor_by_name(
                request, "face_det_thres").as_numpy()[0]
            face_bbox_area_thres = get_input_tensor_by_name(
                request, "face_bbox_area_thres").as_numpy()[0]
            input_image = get_input_tensor_by_name(
                request, "images").as_numpy()[0]
            # reverse the preprocessing done by the yolov5 module
            input_image = np.transpose(input_image, (1, 2, 0))  # CHW -> HWC
            input_image *= 255.0                                # denorm

            outputx = conv_strides_to_anchors(
                [stride_8, stride_16, stride_32], "cpu")
            detections = w_non_max_suppression(
                outputx, num_classes=1, conf_thres=0.4, nms_thres=0.3)

            mow, moh = self.model_face_out_size
            # offsets for faces if required
            tx, ty = 0, 0
            bx, by = 0, 0
            h, w = input_image.shape[:2]
            detections = detections[0]
            face_list = []
            face_bbox_list = []
            face_conf_np_list = []

            # if no faces are detected
            if detections is None:
                face_np_list = np.zeros((1, 3, moh, mow))
                face_bbox_np_list = np.asarray([[0, 0, 0, 0]], dtype=np.int32)
                face_conf_np_list = np.asarray([[0.]], dtype=np.float32)
            # if faces are detected
            else:
                input_image = input_image[..., ::-1]  # RGB2BGR
                # filter weak face detections
                detections = detections[detections[..., 4] > face_det_thres]

                # only select bboxes with area greater than face_bbox_area_thres of total area of frame
                total_area = w * h
                bbox_area = ((detections[:, 2] - detections[:, 0])
                             * (detections[:, 3] - detections[:, 1]))
                bbox_area_perc = 100 * bbox_area / total_area
                detections = detections[bbox_area_perc > face_bbox_area_thres]

                boxs = detections[..., :4]
                # copy faces from image
                for i, box in enumerate(boxs):
                    xmin, ymin, xmax, ymax = map(int, box)
                    x, y, xw, yh = xmin + tx, ymin + ty, xmax + bx, ymax + by
                    x, y, xw, yh = max(x, 0), max(y, 0), min(xw, w), min(yh, h)

                    # preprocessing for face embedding extraction
                    # .copy() only keeps crops in memory
                    face = cv2.resize(
                        input_image[y:yh, x:xw].copy(), (mow, moh)).astype(np.float32)
                    # set to range [-1, 1]
                    face = (face - 127.5) / 127.5
                    face = np.transpose(face, (2, 0, 1))  # HWC to CHW
                    face_list.append(face)
                    face_bbox_list.append(np.asarray(
                        [x, y, xw, yh], dtype=np.int32))
                face_np_list = np.asarray(face_list)
                face_bbox_np_list = np.asarray(face_bbox_list)
                face_conf_np_list = np.asarray(detections[..., 4])

            det_faces_tensor = Tensor("face_detector_faces",
                                      face_np_list.astype(self.face_dtype))
            det_bboxs_tensor = Tensor("face_detector_bboxes",
                                      face_bbox_np_list.astype(self.bbox_dtype))
            det_confs_tensor = Tensor("face_detector_confs",
                                      face_conf_np_list.astype(self.conf_dtype))
            inference_response = InferenceResponse(
                output_tensors=[det_faces_tensor, det_bboxs_tensor, det_confs_tensor])
            responses.append(inference_response)
        return responses
