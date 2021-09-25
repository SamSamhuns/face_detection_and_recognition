# Face Detection

Face detection with mtcnn and cv2 dnn

## cv2 DNN face detection

```shell
$ python detect_face_opencv_dnn.py -w                # webcam mode
$ python detect_face_opencv_dnn.py -i <PATH_TO_IMG>  # image mode
$ python detect_face_opencv_dnn.py -v <PATH_TO_VID>  # video mode
```

## mtcnn face detection

```shell
$ python detect_face_opencv_mtcnn.py -w               # webcam mode
$ python detect_face_opencv_mtcnn.py -i <PATH_TO_IMG> # image mode
$ python detect_face_opencv_mtcnn.py -v <PATH_TO_VID> # video mode
```

## blazeface face detection

```shell
$ python detect_face_blazeface.py -w               # webcam mode
$ python detect_face_blazeface.py -i <PATH_TO_IMG> # image mode
$ python detect_face_blazeface.py -v <PATH_TO_VID> # video mode
```

## Face Extraction from dataset

```shell
$ python extract_faces_img_dataset.py -rd <RAW_DATA_PATH> -td <TARGET_DATA_PATH>
```
