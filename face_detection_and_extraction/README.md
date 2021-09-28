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

## Face Extraction from dataset

```shell
$ python extract_faces_img_dataset.py -rd <RAW_DATA_PATH> -td <TARGET_DATA_PATH>
```
