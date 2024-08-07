import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from modules.utils.image import calculate_bbox_iou
from modules.utils.inference import get_dets_bboxes_confs_lmarks_areas


def test_blank_jpg(mock_opencv_dnn_face_age_gender_model, mock_0_faces_image):
    model = mock_opencv_dnn_face_age_gender_model
    fpath, _ = mock_0_faces_image

    image = cv2.imread(fpath)
    h, w = image.shape[:2]
    iw, ih = model.input_size

    dets, opt_labels = model(image)
    post_dets = get_dets_bboxes_confs_lmarks_areas(
        dets, (w, h), (iw, ih), model.det_thres, model.bbox_area_thres, opt_labels)

    assert len(post_dets.boxes) == 0
    assert len(post_dets.bbox_confs) == 0
    assert len(post_dets.bbox_areas) == 0
    assert len(post_dets.bbox_labels) == 0


def test_3_faces_jpg(mock_opencv_dnn_face_age_gender_model, mock_3_faces_image):
    model = mock_opencv_dnn_face_age_gender_model
    fpath, _ = mock_3_faces_image

    image = cv2.imread(fpath)
    h, w = image.shape[:2]
    iw, ih = model.input_size

    dets, opt_labels = model(image)
    post_dets = get_dets_bboxes_confs_lmarks_areas(
        dets, (w, h), (iw, ih), model.det_thres, model.bbox_area_thres, opt_labels)

    gt_areas = np.array([
        0.02843899, 0.00795599, 0.00623424
    ])
    gt_boxes = np.array([
        [513, 203, 634, 365],
        [408, 213, 469, 299],
        [285, 231, 342, 307],
    ], dtype=np.float32)
    gt_gender_labels = ["Male", "Male", "Male"]
    gt_age_labels = ["38-43", "38-43", "25-32"]

    # Check that IoU with GT if reasonable
    pred_boxes = np.array(post_dets.boxes, dtype=np.float32)
    assert len(gt_boxes) == len(pred_boxes)

    ious = [calculate_bbox_iou(a, b) for a, b in zip(gt_boxes, pred_boxes)]
    iou_mat = np.zeros([len(gt_boxes), len(gt_boxes)])
    np.fill_diagonal(iou_mat, ious)
    gt_idxs, pred_idxs = linear_sum_assignment(-iou_mat)
    is_kept = iou_mat[gt_idxs, pred_idxs] >= 0.8

    assert gt_idxs[is_kept].shape[0] == gt_boxes.shape[0]
    assert np.allclose(gt_areas, post_dets.bbox_areas, atol=0.001)
    assert all([pred_gender in gt_label for pred_gender, gt_label in zip(gt_gender_labels, opt_labels)])
    assert all([pred_age in gt_label for pred_age, gt_label in zip(gt_age_labels, opt_labels)])
