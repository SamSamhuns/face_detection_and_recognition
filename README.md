# Face Detection with OpenCV Caffemodels, MTCNN, Blazeface, and YOLOv5-face

[![Python 3.11](https://img.shields.io/badge/python-3.11-green.svg)](https://www.python.org/downloads/release/python-3110/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/07b06636035d460c8e6e53a6eb88eea4)](https://www.codacy.com/gh/SamSamhuns/face_detection_and_recognition/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=SamSamhuns/face_detection_and_recognition&amp;utm_campaign=Badge_Grade)
[![tests](https://github.com/SamSamhuns/face_detection_and_recognition/actions/workflows/main.yml/badge.svg)](https://github.com/SamSamhuns/face_detection_and_recognition/actions/workflows/main.yml)

<img src="readme_img/detected_faces.jpg" />

- [Face Detection with OpenCV Caffemodels, MTCNN, Blazeface, and YOLOv5-face](#face-detection-with-opencv-caffemodels-mtcnn-blazeface-and-yolov5-face)
  - [Setup](#setup)
  - [Face Detection Models](#face-detection-models)
  - [Face Age and Gender Estimation Models](#face-age-and-gender-estimation-models)
  - [Face Detection and Extraction](#face-detection-and-extraction)
    - [Face detection and extraction weights download](#face-detection-and-extraction-weights-download)
  - [Similar Face Filtering for faces of the same person](#similar-face-filtering-for-faces-of-the-same-person)
    - [Similar face filtering weights download](#similar-face-filtering-weights-download)
    - [Acknowledgements](#acknowledgements)

## Setup

- Option 1. Using `poetry` (Recommended):

```bash
# Install all deps
poetry install --all-groups
# Install specific deps
poetry install --with mobile-facenet,pytorch,onnx
poetry install --with blazeface,pytorch,onnx
poetry install --with mtcnn,tensorflow
# opencv deps are present in the default installation
poetry install --with openvino
poetry install --with yolov5-face,pytorch,onnx
```

Note: After updating `pyproject.toml/poetry.lock`, run `python face_detection_and_extraction/scripts/export_poetry_to_reqs.py` to update requirement files with the updated library versions frompoetry.

- Option 2. Inside a `conda` or `venv` virtual environment:

```bash
pip install --upgrade pip
# install requirements for all face detection
pip install -r requirements.txt  # required for face_extraction
# install model specific requirements
pip install -r face_detection_and_extraction/requirements/mobile_facenet.txt
pip install -r face_detection_and_extraction/requirements/blazeface.txt
pip install -r face_detection_and_extraction/requirements/mtcnn.txt
pip install -r face_detection_and_extraction/requirements/opencv.txt
pip install -r face_detection_and_extraction/requirements/openvino.txt
pip install -r face_detection_and_extraction/requirements/yolov5-face.txt
```

Download the model weights using the instructions [below](#face-detection-and-extraction-weights-download).

## Face Detection Models

CPU Performance recorded on a MacBook Pro with a **2.4 GHz 8-Core Intel Core i9** processor and **16 GB 2400 MHz DDR4** memory with no intensive programs running in the background on a video (**original resolution 576x1024**) with two detectable faces.

| Model                   |         FPS          | <center>Types</center>                                                |         <center>Supported</center>          |
| :---------------------- | :------------------: | :-------------------------------------------------------------------- | :-----------------------------------------: |
| blazeface               | 21 <br/> 16 <br/> 30 | front-camera pytorch <br/> back-camera pytorch <br/> back-camera onnx | :white_check_mark: <br/> :white_check_mark: |
| mtcnn                   |          2           | mtcnn from facenet                                                    |             :white_check_mark:              |
| opencv face-detection   |     18 <br/> 19      | caffemodel <br/> tensorflow graphdef                                  | :white_check_mark: <br/> :white_check_mark: |
| openvino face-detection |     25 <br/> 28      | MobileNetV2 + multiple SSD <br/> SqueezeNet light + single SSD        | :white_check_mark: <br/> :white_check_mark: |
| yolov5-face             |     13 <br/>  13     | yolov5s <br/> yolov5n                                                 | :white_check_mark: <br/> :white_check_mark: |
| arcface                 |         TODO         | arcface                                                               |            :white_large_square:             |

## Face Age and Gender Estimation Models

Performance recorded with same parameters as face-detection above.

| Model  | FPS  | Types                |     Supported      |
| :----- | :--- | :------------------- | :----------------: |
| opencv | 12   | Age and Gender Model | :white_check_mark: |

## Face Detection and Extraction

Instructions inside [face_detection_and_extraction](face_detection_and_extraction/README.md) directory for face detection in images, video, and webcam feed along with face extraction from a dataset of images.

### Face detection and extraction weights download

Download `weights.zip` and unzip weights using `gdown` or directly from this [Google Drive link](https://drive.google.com/file/d/17FXIcOSaVwvpjsnfenkm1bZNmmG6VBIi/view?usp=sharing)

```shell
pip install gdown
gdown 13x471ZMBEWdcagjVgE8C-UJ-c9OjC7NY
unzip weights.zip -d face_detection_and_extraction/
rm weights.zip
```

Or, download weights individually from the GitHub.

```shell
wget https://github.com/SamSamhuns/face_detection_and_recognition/releases/download/v2.0.0/weights.zip -O face_detection_and_extraction/weights.zip
unzip face_detection_and_extraction/weights.zip -d face_detection_and_extraction/
rm face_detection_and_extraction/weights.zip
```

## Similar Face Filtering for faces of the same person

Extract faces from a face dataset that are similiar to a reference face dataset for cleaning face data. Instructions inside the [similar_face_filtering](face_detection_and_recognition/similar_face_filtering/README.md) directory readme.

### Similar face filtering weights download

Download weights from GitHub.

```shell
mkdir -p similar_face_filtering/models/facenet/
wget https://github.com/SamSamhuns/face_detection_and_recognition/releases/download/v2.0.0/facenet_keras_p38.zip -O similar_face_filtering/models/facenet/facenet_keras_p38.zip
unzip similar_face_filtering/models/facenet/facenet_keras_p38.zip -d similar_face_filtering/models/facenet/
rm similar_face_filtering/models/facenet/facenet_keras_p38.zip
```

### Acknowledgements

-   [YOLOv5-face](https://github.com/deepcam-cn/yolov5-face)
-   [Face Age Gender Training](https://github.com/tae898/age-gender)
-   [learnopencv age and gender models](https://github.com/spmallick/learnopencv)
-   [mtcnn](https://github.com/ipazc/mtcnn)
-   [tf-mtcnn](https://github.com/blaueck/tf-mtcnn)
-   [blazeface-python](https://github.com/hollance/BlazeFace-PyTorch)
-   [openvino-open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo)
-   [image orientation correction with DNNs](https://d4nst.github.io/2017/01/12/image-orientation/)
-   [mobile facenet](https://github.com/xuexingyu24/MobileFaceNet_Tutorial_Pytorch)
-   [face recognition embedding](https://github.com/deepinsight/insightface/tree/master/model_zoo)
-   [face anti-spoofing](https://github.com/kprokofi/light-weight-face-anti-spoofing)
