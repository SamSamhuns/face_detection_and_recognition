# Face Detection

[![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)](https://www.python.org/downloads/release/python-390/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg)](https://www.python.org/downloads/release/python-3100/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-green.svg)](https://www.python.org/downloads/release/python-3110/)

Face detection with different models. Use `-h` flag for help. Requirements must be installed beforehand. Note if the webcam mode is not working, make sure the terminal has permission to access the webcam.

## blazeface face detection

```shell
python detect_face_blazeface.py                     # webcam mode default
python detect_face_blazeface.py -i PATH_TO_IMG/VID  # image or video mode
```

## mtcnn face detection

```shell
python detect_face_mtcnn.py                     # webcam mode default
python detect_face_mtcnn.py -i PATH_TO_IMG/VID  # image or video mode
```

## cv2 DNN face detection

```shell
python detect_face_opencv_dnn.py                     # webcam mode default
python detect_face_opencv_dnn.py -i PATH_TO_IMG/VID  # image or video mode
```

## OpenVINO face detection

```shell
python detect_face_openvino.py                     # webcam mode default
python detect_face_openvino.py -i PATH_TO_IMG/VID  # image or video mode
```

## YOLOv5 face detection

```shell
python detect_face_yolov5_face.py                     # webcam mode
python detect_face_yolov5_face.py -i PATH_TO_IMG/VID  # image or video mode
```

# Face Detection model evaluation

## Evaluation on WIDERFACE dataset

### Install pycocotools

`pycocotools` is required for evaluation script.

Inside a virtual env.

```shell
pip install Cython
# make and install pycocotools
git clone https://github.com/cocodataset/cocoapi
cd coco/PythonAPI
make
pip install .
```

### Download WIDER_val split and run evaluation

Download `wider_face_split` and `WIDER_val` from <http://shuoyang1213.me/WIDERFACE/> and place them inside `eval` directory.

```shell
# by default run wider_face_val_split with opencv face detection
PYTHONPATH=$PYTHONPATH:./ python eval/eval_face_detector.py eval/wider_face_split/wider_face_val_bbx_gt.txt eval/WIDER_val/images/
# run wider_face_val_split with yolov5s face detection
PYTHONPATH=$PYTHONPATH:./ python eval/eval_face_detector.py eval1/wider_face_split/wider_face_val_bbx_gt.txt eval1/WIDER_val/images/ --model_type yolov5_face --model weights/yolov5s/yolov5s-face.pt
```

# Face Extraction from Dataset

Extract face images in bulk from an image dataset.

## Face Extraction from imdb-wiki dataset

Download the imdb-wiki data from [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) and place it inside the `face_detection_and_extraction/data` directory.

```shell
python face_extraction/extract_and_clean_imdb_wiki_faces.py -d {data/imdb, data/wiki}
```

## Face Extraction from dataset

```shell
python face_extraction/extract_faces_from_dataset.py --rd SRC_DATA_DIR --td TARGET_DATA_DIR
python face_extraction/extract_faces_from_dataset.py --rd data --td extracted_faces  # using existing imgs in data
```

**Raw Data Organization:**

    SRC_DATA_DIR
                |_ class_1
                         |_ img1/vid1
                         |_ img2/vid2
                         |_ ....
                |_ class_2
                         |_ img1/vid1
                         |_ img2/vid2
                         |_ ....
                ...

## Face Extraction and Labelling from dataset

```shell
python face_extraction/extract_and_label_faces_from_dataset.py --rd SRC_DATA_DIR --td TARGET_DATA_PATH
```

**Annotation data is also saved in the following pickle format:**

    {
      "media_id": "0123456789",
      "media_url": "http://example/0123456789",
      "class_name": "class_1",
      "frame_info": [{
          "frame_num": 0,
          "time_sec": 0,
          "face_ids": [0, 1],
          "face_bboxes": [
            [1, 2, 3, 4],
            [5, 6, 7, 8]
          ],
          "confs": [0.6, 0.8],
          "ages": ["20-50", "20-50"],
          "genders": ["male", "female"],
          "normed_embedding": ["[0.1, ..., 0.2]", "[0.3, ..., 0.4]"]
        },
        {
          "frame_num": 1,
          "time_sec": 1,
          "face_ids": [0, 1],
          "face_bboxes": [
            [1, 2, 3, 4],
            [5, 6, 7, 8]
          ],
          "confs": [0.8, 0.9],
          "ages": ["0-5", "20-50"],
          "genders": ["male", "female"],
          "normed_embedding": ["[0.1, ..., 0.2]", "[0.3, ..., 0.4]"]
        }
      ]
    }

# For Developers

## Running Tests

Install requirements

```shell
pip install requirements/dev.txt
```

Run tests from inside the `face_detection_and_extraction directory`

```shell
python -m pytest tests                 # run pytest
python -m pytest --cov=modules tests/  # run pytest with coverage
```
