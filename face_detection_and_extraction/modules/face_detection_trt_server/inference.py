import sys
import time
import numpy as np
from python_on_whales import docker

sys.path.append(".")
from modules.face_detection_trt_server.triton_utils import FlagConfig, get_client_and_model_metadata_config
from modules.face_detection_trt_server.triton_utils import parse_model_grpc, get_inference_responses
from modules.yolov5_face.onnx.onnx_utils import preprocess_image, get_bboxes_confs_areas
from modules.common_utils import draw_bbox_on_image


class TritonServerInferenceSession(object):

    __slots__ = ["FLAGS", "trt_inf_data", "container"]

    def __init__(self,
                 face_det_thres,
                 face_bbox_area_thres,
                 model_name="ensemble_yolov5_face",
                 container_name="face_det",
                 model_ping_retries=100,
                 device="gpu"):
        if device != "gpu":
            raise ValueError(
                f"{device} device mode is not supported. Only `gpu` mode is supported")
        self.FLAGS = FlagConfig()
        self.FLAGS.face_det_thres = face_det_thres
        self.FLAGS.face_bbox_area_thres = face_bbox_area_thres
        self.FLAGS.model_name = model_name
        self.FLAGS.model_version = ""  # empty str means use latest
        self.FLAGS.protocol = "grpc"
        self.FLAGS.url = '127.0.0.1:8081'
        self.FLAGS.verbose = False
        self.FLAGS.classes = 0  # classes must be set to 0
        self.FLAGS.batch_size = 1
        self.FLAGS.fixed_input_width = 640
        self.FLAGS.fixed_input_height = 640

        # build required docker container
        docker.build("modules/face_detection_trt_server/",
                     tags="yolov5_face_detection:latest")
        # kill triton-server docker container if it already exists
        if docker.container.exists(container_name):
            docker.container.remove(container_name, force=True)
            print(f"Stopping and removing container {container_name}")
        else:
            print(f"Container {container_name} is not running")
        # start triton-server docker container
        self.container = docker.run(image="yolov5_face_detection:latest", name=container_name,
                                    shm_size='1g', ulimit=['memlock=-1', 'stack=67108864'],
                                    gpus='3', detach=True, publish=[(8081, 8081)])
        # wait for container to start
        for _ in range(model_ping_retries):
            model_info = get_client_and_model_metadata_config(self.FLAGS)
            if model_info == -1:  # error getting model info
                for i in range(5):
                    print(f"Model not ready. Reattempting after {5-i} sec...")
                    time.sleep(1)
            else:
                break
        if model_info == -1:  # error getting model info
            raise Exception("Model could not start in time")

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

        # if a model with only one input, remaining two inputs are ignored
        image_data_list = [image_data,
                           np.array([inf_sess.FLAGS.face_det_thres],
                                    dtype=np.float32),
                           np.array([inf_sess.FLAGS.face_bbox_area_thres],
                                    dtype=np.float32)]
        # get one inference result
        detections = get_inference_responses(
            image_data_list, inf_sess.FLAGS, inf_sess.trt_inf_data)[0]

        # if no faces were detected
        # a dummy bounding box [[0,0,0,0]] is returned by model
        zeros_bbox = np.asarray([[0, 0, 0, 0]], dtype=np.int32)
        boxes = detections.as_numpy(
            "ENSEMBLE_FACE_DETECTOR_BBOXES").astype('int')
        if len(boxes) == 0 or (len(boxes) == 1 and (boxes == zeros_bbox).all()):
            return None
        bbox_confs = detections.as_numpy(
            "ENSEMBLE_FACE_DETECTOR_CONFS")
        bbox_confs = bbox_confs.reshape(len(bbox_confs), 1)
        detection_result = np.concatenate([boxes, bbox_confs], axis=1)
        return detection_result


if __name__ == "__main__":
    import cv2
    inf = TritonServerInferenceSession(0.7, 0.10)
    cv2_image = cv2.imread(sys.argv[1])
    out = inf.inference_trt_model_yolov5_face(inf, cv2_image, (640, 640))
    print("detections out shape", out.shape)
    oh, ow = cv2_image.shape[:2]
    boxes, bbox_confs, bbox_area_percs = get_bboxes_confs_areas(
        out, 0.7, 0.10, (ow, oh), (640, 640))
    draw_bbox_on_image(cv2_image, boxes, bbox_confs, bbox_area_percs)

    print("boxes", boxes)
    print("bbox_confs", bbox_confs)
    print("bbox_area_percs", bbox_area_percs)
    cv2.imshow("YOLOv5 face", cv2_image)
    cv2.waitKey(0)
