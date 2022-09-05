# This script is used to estimate an accuracy of different face detection models.
# COCO evaluation tool is used to compute an accuracy metrics (Average Precision).
# Script works with different face detection datasets.
import os
import sys
import json
import argparse

import cv2 as cv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detect_face_yolov5_face import load_model as load_yolov5_face

dataset = {}
dataset['images'] = []
dataset['categories'] = [{'id': 0, 'name': 'face'}]
dataset['annotations'] = []


def addImage(imagePath):
    assert('images' in dataset)
    imageId = len(dataset['images'])
    dataset['images'].append({
        'id': int(imageId),
        'file_name': imagePath
    })
    return imageId


def addBBox(imageId, left, top, width, height):
    assert('annotations' in dataset)
    dataset['annotations'].append({
        'id': len(dataset['annotations']),
        'image_id': int(imageId),
        'category_id': 0,  # Face
        'bbox': [int(left), int(top), int(width), int(height)],
        'iscrowd': 0,
        'area': float(width * height)
    })


def addDetection(detections, imageId, left, top, width, height, score):
    detections.append({
        'image_id': int(imageId),
        'category_id': 0,  # Face
        'bbox': [int(left), int(top), int(width), int(height)],
        'score': float(score)
    })


def populate_ds_with_wider_dataset(annotations, images):
    with open(annotations, 'rt') as f:
        lines = [line.rstrip('\n') for line in f]
        lineId = 0
        while lineId < len(lines):
            # Image
            imgPath = lines[lineId]
            lineId += 1
            imageId = addImage(os.path.join(images, imgPath))

            # Faces
            numFaces = int(lines[lineId])
            lineId += 1
            for i in range(numFaces):
                params = [int(v) for v in lines[lineId].split()]
                lineId += 1
                left, top, width, height = params[0], params[1], params[2], params[3]
                addBBox(imageId, left, top, width, height)


def evaluate():
    cocoGt = COCO('annotations.json')
    cocoDt = cocoGt.loadRes('detections.json')
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def parse_cmd_args():
    parser = argparse.ArgumentParser(
        description='Evaluate face detection algorithms '
        'using COCO evaluation tool, http://cocodataset.org/#detections-eval.'
        'Download WIDER_val and wider_face_split in current directory')
    parser.add_argument(
        'ann', default="eval/wider_face_split/wider_face_val_bbx_gt.txt",
        help="Path to text file with ground truth annotations (default: %(default)s).")
    parser.add_argument(
        'pics', default="eval/WIDER_val/images/",
        help="Path to images root directory (default: %(default)s).")
    parser.add_argument(
        '--model_type', default="opencv", choices=["opencv", "yolov5_face"],
        help="Model type (default: %(default)s).")
    parser.add_argument(
        '--proto', default="weights/face_detection_caffe/deploy.prototxt.txt",
        help="Path to .prototxt of Caffe model or .pbtxt of TensorFlow graph (default: %(default)s).")
    parser.add_argument(
        '--model', default="weights/face_detection_caffe/res10_300x300_ssd_iter_140000.caffemodel",
        help="Path to .caffemodel trained in Caffe or .pb from TensorFlow (default: %(default)s).")
    return parser.parse_args()


def main():
    args = parse_cmd_args()
    # ##################### Convert to COCO annotations format #####################
    populate_ds_with_wider_dataset(args.ann, args.pics)

    with open('annotations.json', 'wt') as f:
        json.dump(dataset, f)

    # ############################ Obtain detections ###############################
    detections = []
    if args.proto and args.model and args.model_type == "opencv":
        net = cv.dnn.readNet(args.proto, args.model)

        def detect(img, imageId):
            net.setInput(cv.dnn.blobFromImage(
                img, 1.0, (300, 300), (104., 177., 123.), False, False))
            out = net.forward()

            for i in range(out.shape[2]):
                confidence = out[0, 0, i, 2]
                left = int(out[0, 0, i, 3] * img.shape[1])
                top = int(out[0, 0, i, 4] * img.shape[0])
                right = int(out[0, 0, i, 5] * img.shape[1])
                bottom = int(out[0, 0, i, 6] * img.shape[0])

                x = max(0, min(left, img.shape[1] - 1))
                y = max(0, min(top, img.shape[0] - 1))
                w = max(0, min(right - x + 1, img.shape[1] - x))
                h = max(0, min(bottom - y + 1, img.shape[0] - y))
                addDetection(detections, imageId, x, y, w, h, score=confidence)
    elif args.model and args.model_type == "yolov5_face":
        # using yolov5-face det model
        net = load_yolov5_face(args.model, 0.1, 0.001, (640, 640), "cpu")

        def detect(img, imageId):
            out = net(img)
            for i in range(out.shape[0]):
                confidence = out[i, -1]
                left = int(out[i, 0] * img.shape[1])
                top = int(out[i, 1] * img.shape[0])
                right = int(out[i, 2] * img.shape[1])
                bottom = int(out[i, 3] * img.shape[0])

                x = max(0, min(left, img.shape[1] - 1))
                y = max(0, min(top, img.shape[0] - 1))
                w = max(0, min(right - x + 1, img.shape[1] - x))
                h = max(0, min(bottom - y + 1, img.shape[0] - y))
                addDetection(detections, imageId, x, y, w, h, score=confidence)

    for i in range(len(dataset['images'])):
        sys.stdout.write('\r%d / %d' % (i + 1, len(dataset['images'])))
        sys.stdout.flush()

        img = cv.imread(dataset['images'][i]['file_name'])
        imageId = int(dataset['images'][i]['id'])
        detect(img, imageId)

    with open('detections.json', 'wt') as f:
        json.dump(detections, f)

    evaluate()


if __name__ == "__main__":
    main()

# OpenCV2 DNN model
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.077
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.144
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.072
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.001
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.182
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.557
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.036
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.082
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.102
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.012
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.259
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.638

# yolov5s
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.211
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.504
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.126
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.121
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.405
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.541
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.045
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.155
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.240
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.144
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.459
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.604
