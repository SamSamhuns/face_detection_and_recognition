# Face Detection

## Face Detection with OpenCV Caffemodels, MTCNN, Blazeface, and YOLOv5-face

<img src="readme_img/detected_faces.jpg" />

## Setup

Inside a virtual environment:

```bash
# install requirements for all face detection models
$ pip install -r requirements.txt
# install model specific requirements for the face detection models
$ pip install -r face_detection_and_extraction/requirements/blazeface.txt
$ pip install -r face_detection_and_extraction/requirements/mtcnn.txt
$ pip install -r face_detection_and_extraction/requirements/opencv.txt
$ pip install -r face_detection_and_extraction/requirements/openvino.txt
$ pip install -r face_detection_and_extraction/requirements/yolov5-face.txt
```

## Face Detection Models Implemented

Performance recorded on a MacBook Pro with a **2.4 GHz 8-Core Intel Core i9** processor and **16 GB 2400 MHz DDR4** memory with no intensive programs running in the background.

| Model                   | FPS  | Types                                                          |                  Supported                  |
| :---------------------- | :--- | :------------------------------------------------------------- | :-----------------------------------------: |
| blazeface               | TODO | front-camera <br/> back-camera                                 | :white_check_mark: <br/> :white_check_mark: |
| mtcnn                   | TODO | mtcnn from facenet                                             |              :white_check_mark:             |
| opencv face-detection   | TODO | caffemodel <br/> tensorflow graphdef                           | :white_check_mark: <br/> :white_check_mark: |
| openvino face-detection | TODO | MobileNetV2 + multiple SSD <br/> SqueezeNet light + single SSD | :white_check_mark: <br/> :white_check_mark: |
| yolov5-face             | TODO | yolov5s <br/> yolov5n                                          | :white_check_mark: <br/> :white_check_mark: |
| arcface                 | TODO | arcface                                                        |                :white_square:               |

### Face Age and Gender Estimation

| Model  | FPS  | Types                |      Supported     |
| :----- | :--- | :------------------- | :----------------: |
| opencv | TODO | Age and Gender Model | :white_check_mark: |

## Face Detection and Extraction

Instructions inside `face_detection_and_extraction` for face detection in images, video, and webcam feed along with face extraction from a dataset of images.

## Face Feat Extraction and Filtering for faces of the same person

Instructions inside `face_feat_extraction_and_filtering` for face feat extraction and face similarity extraction.

### Acknowledgements

-  (YOLOv5-face)[https://github.com/deepcam-cn/yolov5-face]
-  (learnopencv age and gender models)[https://github.com/spmallick/learnopencv]
-  (mtcnn)[https://github.com/ipazc/mtcnn]
-  (blazeface-python)[https://github.com/hollance/BlazeFace-PyTorch]
-  (openvino-open_model_zoo)[https://github.com/openvinotoolkit/open_model_zoo]
