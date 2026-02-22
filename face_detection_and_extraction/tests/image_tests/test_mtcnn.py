import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from modules.utils.image import calculate_bbox_iou
from modules.utils.inference import get_dets_bboxes_confs_lmarks_areas


def _match_preds_to_gt(gt_boxes: np.ndarray, pred_boxes: np.ndarray):
    iou_mat = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)
    for gt_idx, gt_box in enumerate(gt_boxes):
        for pred_idx, pred_box in enumerate(pred_boxes):
            iou_mat[gt_idx, pred_idx] = calculate_bbox_iou(gt_box, pred_box)

    gt_idxs, pred_idxs = linear_sum_assignment(-iou_mat)
    sort_idx = np.argsort(gt_idxs)
    return iou_mat, gt_idxs[sort_idx], pred_idxs[sort_idx]


def test_slow_blank_jpg(mock_mtcnn_slow_model, mock_0_faces_image):
    model = mock_mtcnn_slow_model
    fpath, _ = mock_0_faces_image

    image = cv2.imread(fpath)
    h, w = image.shape[:2]

    dets = model(image)
    post_dets = get_dets_bboxes_confs_lmarks_areas(
        dets, (w, h), (w, h), model.det_thres, model.bbox_area_thres)

    assert len(post_dets.boxes) == 0
    assert len(post_dets.bbox_confs) == 0
    assert len(post_dets.bbox_areas) == 0
    assert len(post_dets.bbox_lmarks) == 0


def test_fast_blank_jpg(mock_mtcnn_fast_model, mock_0_faces_image):
    model = mock_mtcnn_fast_model
    fpath, _ = mock_0_faces_image

    image = cv2.imread(fpath)
    h, w = image.shape[:2]

    dets = model(image)
    post_dets = get_dets_bboxes_confs_lmarks_areas(
        dets, (w, h), (w, h), model.det_thres, model.bbox_area_thres)

    assert len(post_dets.boxes) == 0
    assert len(post_dets.bbox_confs) == 0
    assert len(post_dets.bbox_areas) == 0
    assert len(post_dets.bbox_lmarks) == 0


def test_slow_3_faces_jpg(mock_mtcnn_slow_model, mock_3_faces_image):
    model = mock_mtcnn_slow_model
    fpath, _ = mock_3_faces_image

    image = cv2.imread(fpath)
    h, w = image.shape[:2]

    dets = model(image)
    iw, ih = model.input_size
    post_dets = get_dets_bboxes_confs_lmarks_areas(
        dets, (w, h), (iw, ih), model.det_thres, model.bbox_area_thres)

    gt_areas = np.array([
        0.00979424, 0.01138117, 0.04899691
    ])
    gt_boxes = np.array([
        [285., 235., 341., 303.],
        [409., 216., 468., 291.],
        [506., 209., 633., 359.]
    ], dtype=np.float32)
    gt_lmarks = np.array([
        [302., 260., 329., 263., 316., 274., 300., 283., 326., 287.],
        [420., 246., 449., 243., 431., 260., 424., 277., 448., 275.],
        [537., 267., 596., 263., 560., 296., 543., 324., 597., 323.]
    ], dtype=np.float32)

    # Check that IoU with GT if reasonable
    pred_boxes = np.array(post_dets.boxes, dtype=np.float32)
    assert len(gt_boxes) == len(pred_boxes)

    iou_mat, gt_idxs, pred_idxs = _match_preds_to_gt(gt_boxes, pred_boxes)
    assert gt_idxs.shape[0] == gt_boxes.shape[0]
    assert np.all(iou_mat[gt_idxs, pred_idxs] >= 0.8)

    pred_areas = np.array(post_dets.bbox_areas, dtype=np.float32)[pred_idxs]
    pred_lmarks = np.array(post_dets.bbox_lmarks, dtype=np.float32)[pred_idxs]
    # Slow MTCNN (keras/tf stack) can shift box sizes slightly across environments.
    assert np.allclose(gt_areas, pred_areas, rtol=0.15, atol=0.001)
    assert np.allclose(gt_lmarks, pred_lmarks, atol=1)


def test_fast_3_faces_jpg(mock_mtcnn_fast_model, mock_3_faces_image):
    model = mock_mtcnn_fast_model
    fpath, _ = mock_3_faces_image

    image = cv2.imread(fpath)
    h, w = image.shape[:2]

    dets = model(image)
    iw, ih = model.input_size
    post_dets = get_dets_bboxes_confs_lmarks_areas(
        dets, (w, h), (iw, ih), model.det_thres, model.bbox_area_thres)

    gt_areas = np.array([
        0.01053372, 0.01290631, 0.05193915
    ])
    gt_boxes = np.array([
        [283., 234., 340., 306.],
        [407., 212., 469., 294.],
        [508., 199., 635., 358.]
    ], dtype=np.float32)
    gt_lmarks = np.array([
        [303., 259., 329., 263., 317., 274., 300., 283., 327., 286.],
        [421., 247., 449., 243., 431., 260., 425., 277., 449., 274.],
        [538., 264., 596., 263., 561., 297., 541., 321., 595., 321.]
    ], dtype=np.float32)

    # Check that IoU with GT if reasonable
    pred_boxes = np.array(post_dets.boxes, dtype=np.float32)
    assert len(gt_boxes) == len(pred_boxes)

    iou_mat, gt_idxs, pred_idxs = _match_preds_to_gt(gt_boxes, pred_boxes)
    assert gt_idxs.shape[0] == gt_boxes.shape[0]
    assert np.all(iou_mat[gt_idxs, pred_idxs] >= 0.8)

    pred_areas = np.array(post_dets.bbox_areas, dtype=np.float32)[pred_idxs]
    pred_lmarks = np.array(post_dets.bbox_lmarks, dtype=np.float32)[pred_idxs]
    assert np.allclose(gt_areas, pred_areas, atol=0.001)
    assert np.allclose(gt_lmarks, pred_lmarks, atol=1)
