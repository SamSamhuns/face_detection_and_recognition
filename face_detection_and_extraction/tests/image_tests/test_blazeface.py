import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from modules.utils.image import calculate_bbox_iou
from modules.models.inference import get_dets_bboxes_confs_lmarks_areas


def test_torch_blank_jpg(mock_blazeface_torch_model, mock_0_faces_image):
    model = mock_blazeface_torch_model
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


def test_onnx_blank_jpg(mock_blazeface_onnx_model, mock_0_faces_image):
    model = mock_blazeface_onnx_model
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


def test_torch_3_faces_jpg(mock_blazeface_torch_model, mock_3_faces_image):
    model = mock_blazeface_torch_model
    fpath, _ = mock_3_faces_image

    image = cv2.imread(fpath)
    h, w = image.shape[:2]
    iw, ih = model.input_size

    dets = model(image)
    post_dets = get_dets_bboxes_confs_lmarks_areas(
        dets, (w, h), (iw, ih), model.det_thres, model.bbox_area_thres)

    gt_areas = np.array([
        0.00827127, 0.00629996, 0.03269695
    ])
    gt_boxes = np.array([
        [409., 228., 475., 293.],
        [285., 248., 343., 305.],
        [510., 232., 640., 362.]
    ], dtype=np.float32)

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


def test_onnx_3_faces_jpg(mock_blazeface_onnx_model, mock_3_faces_image):
    model = mock_blazeface_onnx_model
    fpath, _ = mock_3_faces_image

    image = cv2.imread(fpath)
    h, w = image.shape[:2]
    iw, ih = model.input_size

    dets = model(image)
    post_dets = get_dets_bboxes_confs_lmarks_areas(
        dets, (w, h), (iw, ih), model.det_thres, model.bbox_area_thres)

    gt_areas = np.array([
        0.00827127, 0.00629996, 0.03269695
    ])
    gt_boxes = np.array([
        [409., 228., 475., 293.],
        [285., 248., 343., 305.],
        [510., 232., 640., 362.]
    ], dtype=np.float32)

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
