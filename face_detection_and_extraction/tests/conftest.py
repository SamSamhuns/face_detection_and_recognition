# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import cv2
import pytest

from modules.openvino.utils import OVNetwork
from detect_face_opencv_dnn import Net as OpenCVDnnNet


def _get_img_content(fpath, mode="rb"):
    with open(fpath, mode) as f:
        img_content = f.read()
    return img_content


# ########################### model mocks #######################################


DET_THRES = 0.70
BBOX_AREA_THRES = 0.12


@pytest.fixture(scope="session")
def mock_openvino_model():
    model_xml_path = "weights/face_detection_0204/model.xml"
    model_bin_path = "weights/face_detection_0204/model.bin"
    model = OVNetwork(model_xml_path, model_bin_path,
                      det_thres=DET_THRES, bbox_area_thres=BBOX_AREA_THRES,
                      device="CPU")
    return model


@pytest.fixture(scope="session")
def mock_opencv_dnn_model():
    prototxt_path = "weights/face_detection_caffe/deploy.prototxt.txt"
    model_path = "weights/face_detection_caffe/res10_300x300_ssd_iter_140000.caffemodel"
    face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    model_in_size = (300, 300)
    model = OpenCVDnnNet(face_net, DET_THRES, BBOX_AREA_THRES, model_in_size)
    return model

# ########################### image mocks #######################################


@pytest.fixture(scope="session")
def mock_0_faces_image():
    fpath = "data/TEST/test1_faces_0.jpg"
    return fpath, _get_img_content(fpath)


@pytest.fixture(scope="session")
def mock_3_faces_image():
    fpath = "data/TEST/test2_faces_3.jpg"
    return fpath, _get_img_content(fpath)
