# Face Detection

Face detection with different networks.

## blazeface face detection

```shell
$ python detect_face_blazeface.py -w               # webcam mode
$ python detect_face_blazeface.py -i <PATH_TO_IMG> # image mode
$ python detect_face_blazeface.py -v <PATH_TO_VID> # video mode
```

## mtcnn face detection

```shell
$ python detect_face_opencv_mtcnn.py -w               # webcam mode
$ python detect_face_opencv_mtcnn.py -i <PATH_TO_IMG> # image mode
$ python detect_face_opencv_mtcnn.py -v <PATH_TO_VID> # video mode
```

## cv2 DNN face detection

```shell
$ python detect_face_opencv_dnn.py -w                # webcam mode
$ python detect_face_opencv_dnn.py -i <PATH_TO_IMG>  # image mode
$ python detect_face_opencv_dnn.py -v <PATH_TO_VID>  # video mode
```

## OpenVINO face detection

```shell
$ python detect_face_openvino.py -w                # webcam mode
$ python detect_face_openvino.py -i <PATH_TO_IMG>  # image mode
$ python detect_face_openvino.py -v <PATH_TO_VID>  # video mode
```

## YOLOv5 face detection

```shell
$ python detect_face_yolov5_face.py -w                # webcam mode
$ python detect_face_yolov5_face.py -i <PATH_TO_IMG>  # image mode
$ python detect_face_yolov5_face.py -v <PATH_TO_VID>  # video mode
```

### Face Extraction Dataset Organization

```
#################### Raw Data Organization #########################
       dataset
             |_ class_1
                       |_ img1/vid1
                       |_ img2/vid2
                       |_ ....
             |_ class_2
                       |_ img1/vid1
                       |_ img2/vid2
                       |_ ....
             ...

example raw data path    = "raw_data/dataset"
example target data path = "target_data/dataset"
###################################################################
```

## Face Extraction from dataset

```shell
$ python data_extraction/extract_faces_from_dataset.py -rd <RAW_DATA_PATH> -td <TARGET_DATA_PATH>
```

## Face Extraction and Labelling from dataset

```shell
$ python data_extraction/extract_and_label_faces_from_dataset.py -rd <RAW_DATA_PATH> -td <TARGET_DATA_PATH>
```

Annotation data is also saved in the following pickle format:

```
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
```

## Face Extraction from imdb-wiki dataset

```shell
$ python data_extraction/extract_and_clean_imdb_wiki_faces.py -d {data/imdb, data/wiki}
```
