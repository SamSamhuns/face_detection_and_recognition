import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from modules.utils.image import calculate_bbox_iou
from modules.utils.inference import get_dets_bboxes_confs_lmarks_areas


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
    iw, ih = model.input_size

    dets = model(image)
    post_dets = get_dets_bboxes_confs_lmarks_areas(
        dets, (w, h), (iw, ih), model.det_thres, model.bbox_area_thres)

    gt_areas = np.array([
        0.00979424, 0.01138117, 0.04899691
    ])
    gt_boxes = np.array([
        [285., 210., 341., 327.],
        [409., 177., 468., 306.],
        [506., 165., 633., 424.]
    ], dtype=np.float32)
    gt_lmarks = np.array([
        [302., 253., 329., 258., 316., 277., 300., 292., 326., 299.],
        [420., 229., 449., 223., 431., 253., 424., 282., 448., 279.],
        [537., 265., 596., 258., 560., 315., 543., 363., 597., 361.]
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
    assert np.allclose(gt_lmarks, post_dets.bbox_lmarks, atol=1)


def test_fast_3_faces_jpg(mock_mtcnn_fast_model, mock_3_faces_image):
    model = mock_mtcnn_fast_model
    fpath, _ = mock_3_faces_image

    image = cv2.imread(fpath)
    h, w = image.shape[:2]
    iw, ih = model.input_size

    dets = model(image)
    post_dets = get_dets_bboxes_confs_lmarks_areas(
        dets, (w, h), (iw, ih), model.det_thres, model.bbox_area_thres)

    gt_areas = np.array([
        0.01053372, 0.01290631, 0.05193915
    ])
    gt_boxes = np.array([
        [283., 208., 340., 333.],
        [407., 169., 469., 311.],
        [508., 148., 635., 422.]
    ], dtype=np.float32)
    gt_lmarks = np.array([
        [303., 252., 329., 257., 317., 277., 300., 292., 327., 297.],
        [421., 230., 449., 224., 431., 253., 425., 281., 449., 277.],
        [538., 260., 596., 258., 561., 316., 541., 359., 595., 358.]
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
    assert np.allclose(gt_lmarks, post_dets.bbox_lmarks, atol=1)
