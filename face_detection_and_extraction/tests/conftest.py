import pytest

from modules.openvino.model import OVModel
from modules.blazeface.model import BlazeFaceModel
from detect_face_mtcnn import load_model as load_MTCNNModel
from detect_face_yolov5_face import load_model as load_YOLOV5FaceModel
from detect_face_opencv_dnn import load_model as load_OpenCVFaceDetModel
from detect_face_opencv_age_gender import load_model as load_OpenCVFaceAgeGenderModel


def _get_img_content(fpath: str, mode: str = "rb"):
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
def mock_opencv_dnn_face_det_model():
    model_path = "weights/face_detection_caffe/res10_300x300_ssd_iter_140000.caffemodel"
    prototxt_path = "weights/face_detection_caffe/deploy.prototxt.txt"
    model_in_size = (300, 400)
    model = load_OpenCVFaceDetModel(
        model_path, prototxt_path, DET_THRES, BBOX_AREA_THRES, model_in_size)
    return model


@pytest.fixture(scope="session")
def mock_opencv_dnn_face_age_gender_model():
    facedet_model = "weights/face_detection_caffe/res10_300x300_ssd_iter_140000.caffemodel"
    facedet_proto = "weights/face_detection_caffe/deploy.prototxt.txt"
    age_proto = "weights/age_net_caffe/age_deploy.prototxt"
    age_model = "weights/age_net_caffe/age_net.caffemodel"
    gender_proto = "weights/gender_net_caffe/gender_deploy.prototxt"
    gender_model = "weights/gender_net_caffe/gender_net.caffemodel"
    model_in_size = (300, 400)
    model = load_OpenCVFaceAgeGenderModel(
        facedet_model, facedet_proto, age_proto, age_model, gender_proto, gender_model,
        DET_THRES, BBOX_AREA_THRES, model_in_size, "cpu")
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


@pytest.fixture(scope="session")
def mock_mtcnn_fast_model():
    model = load_MTCNNModel("fast", DET_THRES, BBOX_AREA_THRES, "cpu")
    return model


@pytest.fixture(scope="session")
def mock_mtcnn_slow_model():
    model = load_MTCNNModel("slow", DET_THRES, BBOX_AREA_THRES, "cpu")
    return model


@pytest.fixture(scope="session")
def mock_yolov5_face_torch_model():
    model_path = "weights/yolov5s/yolov5s-face.pt"
    model = load_YOLOV5FaceModel(model_path, DET_THRES, BBOX_AREA_THRES, (640, 640), "cpu")
    return model


@pytest.fixture(scope="session")
def mock_yolov5_face_onnx_model():
    model_path = "weights/yolov5s/yolov5s-face.onnx"
    model = load_YOLOV5FaceModel(model_path, DET_THRES, BBOX_AREA_THRES, (640, 640), "cpu")
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
