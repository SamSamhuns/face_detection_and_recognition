import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from modules.utils.image import calculate_bbox_iou
from modules.utils.inference import get_dets_bboxes_confs_lmarks_areas


def test_blank_jpg(mock_openvino_model, mock_0_faces_image):
    model = mock_openvino_model
    fpath, _ = mock_0_faces_image

    image = cv2.imread(fpath)
    h, w = image.shape[:2]
    iw, ih = model.input_size

    dets = model(image)
    post_dets = get_dets_bboxes_confs_lmarks_areas(
        dets, (w, h), (iw, ih), model.det_thres, model.bbox_area_thres)

    assert len(post_dets.boxes) == 0
    assert len(post_dets.bbox_confs) == 0
    assert len(post_dets.bbox_areas) == 0


def test_3_faces_jpg(mock_openvino_model, mock_3_faces_image):
    model = mock_openvino_model
    fpath, _ = mock_3_faces_image

    image = cv2.imread(fpath)
    h, w = image.shape[:2]
    iw, ih = model.input_size

    dets = model(image)
    post_dets = get_dets_bboxes_confs_lmarks_areas(
        dets, (w, h), (iw, ih), model.det_thres, model.bbox_area_thres)

    gt_areas = np.array([
        0.0099964741, 0.0377225852, 0.0082628834
    ])
    gt_boxes = np.array([
        [408, 213, 469, 299],
        [513, 203, 634, 365],
        [285, 231, 342, 307],
    ], dtype=np.float32)

    # Check that IoU with GT if reasonable
    pred_boxes = np.array(post_dets.boxes, dtype=np.float32)
    assert len(gt_boxes) == len(pred_boxes)

    # Calculate IoUs and form the cost matrix (negative because we want to maximize IoU)
    iou_mat = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou_mat[i, j] = calculate_bbox_iou(gt_box, pred_box)

     # Solve the assignment problem to maximize the sum of IoUs
    gt_idxs, pred_idxs = linear_sum_assignment(-iou_mat)

    # Verify IoUs are above a reasonable threshold, e.g., 0.8
    assert np.all(iou_mat[gt_idxs, pred_idxs] >=0.8), \
        "Some IoUs are below the threshold of 0.8."

    # Check areas with the alignment given by the Hungarian algorithm
    aligned_areas = post_dets.bbox_areas[pred_idxs]
    assert np.allclose(gt_areas, aligned_areas, atol=0.001), \
        "Area close check failed with aligned indices."
