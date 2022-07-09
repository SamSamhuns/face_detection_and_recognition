import cv2
import sys
import time
import numpy as np
from python_on_whales import docker

sys.path.append(".")
from modules.facenet_trt_server.triton_utils import FlagConfig, get_client_and_model_metadata_config
from modules.facenet_trt_server.triton_utils import parse_model_grpc, get_inference_responses
from modules.utils.image import standardize_image


class TritonServerInferenceSession(object):

    __slots__ = ["FLAGS", "trt_inf_data", "container"]

    def __init__(self,
                 model_name="facenet",
                 container_name="facenet_feat_ext",
                 model_ping_retries=100,
                 device="gpu",
                 port=8090):
        if device != "gpu":
            raise ValueError(
                f"{device} device mode is not supported. Only `gpu` mode is supported")
        self.FLAGS = FlagConfig()
        self.FLAGS.model_name = model_name
        self.FLAGS.model_version = ""  # empty str means use latest
        self.FLAGS.protocol = "grpc"
        self.FLAGS.url = f'127.0.0.1:{port}'
        self.FLAGS.verbose = False
        self.FLAGS.classes = 0  # classes must be set to 0
        self.FLAGS.batch_size = 1
        self.FLAGS.fixed_input_width = 160
        self.FLAGS.fixed_input_height = 160

        # build required docker container
        docker.build("modules/facenet_trt_server/",
                     tags="facenet_feat_extraction:latest")
        # kill triton-server docker container if it already exists
        if docker.container.exists(container_name):
            docker.container.remove(container_name, force=True)
            print(f"Stopping and removing container {container_name}")
        else:
            print(f"Container {container_name} is not running")
        # start triton-server docker container
        self.container = docker.run(image="facenet_feat_extraction:latest", name=container_name,
                                    shm_size='1g', ulimit=['memlock=-1', 'stack=67108864'],
                                    gpus='1', detach=True, publish=[(port, 8081)])
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
        max_batch_size, input_name, output_name, h, w, c, format, dtype = parse_model_grpc(
            model_metadata, model_config.config)

        self.trt_inf_data = (triton_client, input_name,
                             output_name, dtype, max_batch_size)

    @staticmethod
    def inference_trt_model_facenet(inf_sess, cv2_image, model_in_size, **kwargs):
        rsz_image = cv2.resize(cv2_image, (model_in_size))
        image_data = np.expand_dims(standardize_image(rsz_image), axis=0)
        if len(image_data) == 0:
            print("Image data is missing. Aborting inference")
            return -1

        # if a model with only one input, remaining two inputs are ignored
        image_data_list = [image_data]
        # get one inference result
        detections = get_inference_responses(
            image_data_list, inf_sess.FLAGS, inf_sess.trt_inf_data)[0]
        features = detections.as_numpy("Bottleneck_BatchNorm")
        return features.squeeze()


if __name__ == "__main__":
    inf = TritonServerInferenceSession()
    cv2_image = cv2.imread(sys.argv[1])
    out = inf.inference_trt_model_facenet(inf, cv2_image, (160, 160))
    print("features out shape", out.shape)
