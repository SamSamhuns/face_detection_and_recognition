# Face Detection

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
$ pip install -r face_detection_and_extraction/requirements/yolov5-face.txt
```

## Face Detection and Extraction

Instructions inside `face_detection_and_extraction` for face detection in images, video, and webcam feed along with face extraction from a dataset of images.

## Face Feat Extraction and Filtering for faces of the same person

Instructions inside `face_feat_extraction_and_filtering` for face feat extraction and face similarity extraction.
