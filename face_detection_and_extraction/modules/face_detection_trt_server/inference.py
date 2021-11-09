import sys
import numpy as np

sys.path.append(".")
from modules.face_detection_trt_server.triton_utils import FlagConfig, get_client_and_model_metadata_config
from modules.face_detection_trt_server.triton_utils import parse_model_grpc, get_inference_responses
from modules.yolov5_face.onnx.onnx_utils import preprocess_image, get_bboxes_confs_areas
from modules.common_utils import draw_bbox_on_image


class TritonServerInferenceSession(object):

    __slots__ = ["model_name", "FLAGS", "trt_inf_data"]

    def __init__(self, model_name="ensemble_yolov5_face"):
        self.model_name = model_name
        self.FLAGS = FlagConfig()
        self.FLAGS.model_name = model_name
        self.FLAGS.model_version = ""  # empty str means use latest
        self.FLAGS.protocol = "grpc"
        self.FLAGS.url = '127.0.0.1:8081'
        self.FLAGS.verbose = False
        self.FLAGS.classes = 0  # classes must be set to 0
        self.FLAGS.batch_size = 1
        self.FLAGS.fixed_input_width = 640
        self.FLAGS.fixed_input_height = 640

        model_info = get_client_and_model_metadata_config(self.FLAGS)
        if model_info == -1:  # error getting model info
            return -1
        triton_client, model_metadata, model_config = model_info

        # input_name, output_name, format, dtype are all lists
        max_batch_size, input_name, output_name, c, h, w, format, dtype = parse_model_grpc(
            model_metadata, model_config.config)

        self.trt_inf_data = (triton_client, input_name,
                             output_name, dtype, max_batch_size)

    @staticmethod
    def inference_trt_model_yolov5_face(inf_sess, cv2_image, model_in_size, **kwargs):
        image_data = preprocess_image(cv2_image, input_size=model_in_size)
        if len(image_data) == 0:
            print("Image data is missing. Aborting inference")
            return -1

        # if a model with only one input, i.e. edetlite4 is used,
        # the remaining two inputs are ignored
        image_data_list = [image_data,
                           np.array([inf_sess.FLAGS.face_det_thres],
                                    dtype=np.float32),
                           np.array([inf_sess.FLAGS.face_bbox_area_thres], dtype=np.float32)]
        # get one inference result
        detections = get_inference_responses(
            image_data_list, inf_sess.FLAGS, inf_sess.trt_inf_data)[0]

        # if no faces were detected
        # a dummy bounding box [0,0,0,0] is returned by model
        zeros_bbox = np.asarray([[0, 0, 0, 0]], dtype=np.int32)
        boxes = detections.as_numpy(
            "ENSEMBLE_FACE_DETECTOR_BBOXES").astype('int')
        if len(boxes) == 1 and (boxes == zeros_bbox).all():
            return None
        bbox_confs = detections.as_numpy(
            "ENSEMBLE_FACE_DETECTOR_CONFS")
        bbox_confs = bbox_confs.reshape(len(bbox_confs), 1)

        print("boxes.shape", boxes.shape)
        print("bbox_confs.shape", bbox_confs.shape)
        detection_result = np.concatenate([boxes, bbox_confs], axis=1)
        return detection_result


if __name__ == "__main__":
    import cv2
    inf = TritonServerInferenceSession()
    cv2_image = cv2.imread(sys.argv[1])
    out = inf.inference_trt_model_yolov5_face(inf, cv2_image, (640, 640))
    print("detections out shape", out.shape)
    oh, ow = cv2_image.shape[:2]
    boxes, bbox_confs, bbox_area_percs = get_bboxes_confs_areas(out, 0.7, 0.10, (ow, oh), (640, 640))
    draw_bbox_on_image(cv2_image, boxes, bbox_confs, bbox_area_percs)

    print("boxes", boxes)
    print("bbox_confs", bbox_confs)
    print("bbox_area_percs", bbox_area_percs)
    cv2.imshow("YOLOv5 face", cv2_image)
    cv2.waitKey(0)
