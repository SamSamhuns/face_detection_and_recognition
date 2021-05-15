# Face Detection

Face detection with mtcnn and cv2 dnn

Set a virtualenv and `pip install -r requirements.txt`

## cv2 DNN face detection

```shell
$ python dnn_face_detect.py -w                # webcam mode
$ python dnn_face_detect.py -i <PATH_TO_IMG>  # image mode
$ python dnn_face_detect.py -v <PATH_TO_VID>  # video mode
```

## mtcnn face detection

```shell
$ python mtcnn_face_detect.py -w               # webcam mode
$ python mtcnn_face_detect.py -i <PATH_TO_IMG> # image mode
$ python mtcnn_face_detect.py -v <PATH_TO_VID> # video mode
```

## Face Extraction from dataset

```shell
$ python extract_faces_img_dataset.py -rd <RAW_DATA_PATH> -td <TARGET_DATA_PATH>
```
