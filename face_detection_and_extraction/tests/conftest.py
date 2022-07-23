import pytest

from modules.openvino.model import OVModel
from modules.blazeface.model import BlazeFaceModel
from detect_face_opencv_dnn import load_model as load_OpenCVModel


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
    model = OVModel(model_xml_path, model_bin_path,
                    det_thres=DET_THRES, bbox_area_thres=BBOX_AREA_THRES,
                    device="CPU")
    return model


@pytest.fixture(scope="session")
def mock_opencv_dnn_model():
    model_path = "weights/face_detection_caffe/res10_300x300_ssd_iter_140000.caffemodel"
    prototxt_path = "weights/face_detection_caffe/deploy.prototxt.txt"
    model_in_size = "300,400"
    model = load_OpenCVModel(model_path, prototxt_path, DET_THRES, BBOX_AREA_THRES, model_in_size)
    return model


@pytest.fixture(scope="session")
def mock_blazeface_torch_model():
    model_path = "weights/blazeface/blazefaceback.pth"
    model = BlazeFaceModel(model_path, DET_THRES, BBOX_AREA_THRES, "back")
    return model


@pytest.fixture(scope="session")
def mock_blazeface_onnx_model():
    model_path = "weights/blazeface/blazefaceback.onnx"
    model = BlazeFaceModel(model_path, DET_THRES, BBOX_AREA_THRES, "back")
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
