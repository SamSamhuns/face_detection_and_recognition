import cv2
import torch
import numpy as np


def _make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def conv_strides_to_anchors(pred, device="cpu"):
    """
    post process when the final layer export is set to True during onnx export
    Converting stride layers to anchors
    """
    # for strides 8、16、32
    stride = torch.tensor([8., 16., 32.]).to(device)

    x = [torch.from_numpy(pred[0]).to(device), torch.from_numpy(
        pred[1]).to(device), torch.from_numpy(pred[2]).to(device)]

    no = 16
    nl = 3
    grid = [torch.zeros(1).to(device)] * nl
    anchor_grid = torch.tensor([[[[[[4.,   5.]]], [[[8.,  10.]]], [[[13.,  16.]]]]],
                                [[[[[23.,  29.]]], [[[43.,  55.]]], [[[73., 105.]]]]],
                                [[[[[146., 217.]]], [[[231., 300.]]], [[[335., 433.]]]]]]).to(device)

    z = []
    for i in range(len(x)):
        bs, ny, nx = x[i].shape[0], x[i].shape[2], x[i].shape[3]
        if grid[i].shape[2:4] != x[i].shape[2:4]:
            grid[i] = _make_grid(nx, ny).to(x[i].device)
        y = torch.full_like(x[i], 0)
        y[..., [0, 1, 2, 3, 4, 15]] = x[i][..., [0, 1, 2, 3, 4, 15]].sigmoid()
        y[..., 5:15] = x[i][..., 5:15]
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 +
                       grid[i].to(x[i].device)) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh

        y[..., 5:7] = y[..., 5:7] * anchor_grid[i] + \
            grid[i].to(x[i].device) * stride[i]  # landmark x1 y1
        y[..., 7:9] = y[..., 7:9] * anchor_grid[i] + \
            grid[i].to(x[i].device) * stride[i]  # landmark x2 y2
        y[..., 9:11] = y[..., 9:11] * anchor_grid[i] + \
            grid[i].to(x[i].device) * stride[i]  # landmark x3 y3
        y[..., 11:13] = y[..., 11:13] * anchor_grid[i] + \
            grid[i].to(x[i].device) * stride[i]  # landmark x4 y4
        y[..., 13:15] = y[..., 13:15] * anchor_grid[i] + \
            grid[i].to(x[i].device) * stride[i]  # landmark x5 y5

        z.append(y.view(bs, -1, no))
    return torch.cat(z, 1)


def w_bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Calculate IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                          0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                          0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
        torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def w_non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    # Find the upper left and lower right corners
    # box_corner = prediction.new(prediction.shape)
    box_corner = torch.FloatTensor(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Use confidence for the first round of screening
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]

        if not image_pred.size(0):
            continue

        # Obtain type and its confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        # content obtained is (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat(
            (image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        # Type of acquisition
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()

        for c in unique_labels:
            # Obtain all prediction results after a certain type of preliminary screening
            detections_class = detections[detections[:, -1] == c]
            # Sort according to the confidence of the existence of the object
            _, conf_sort_index = torch.sort(
                detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Take out this category with the highest confidence, judge step by step, and
                # judge whether the degree of coincidence is greater than nms_thres, and if so, remove it
                max_detections.append(detections_class[0].unsqueeze(0))
                if len(detections_class) == 1:
                    break
                ious = w_bbox_iou(max_detections[-1], detections_class[1:])
                detections_class = detections_class[1:][ious < nms_thres]
            # Stacked
            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output


def pad_resize_image(cv2_img, new_size=(640, 480), color=(125, 125, 125)) -> np.ndarray:
    """
    resize and pad image if necessary, maintaining orig scale
    args:
        cv2_img: numpy.ndarray = cv2 image
        new_size: tuple(int, int) = (width, height)
        color: tuple(int, int, int) = (B, G, R)
    """
    in_h, in_w = cv2_img.shape[:2]
    new_w, new_h = new_size
    # rescale down
    scale = min(new_w / in_w, new_h / in_h)
    # get new sacled widths and heights
    scl_new_w, scl_new_h = int(in_w * scale), int(in_h * scale)
    rsz_img = cv2.resize(cv2_img, (scl_new_w, scl_new_h))
    # calculate deltas for padding
    d_w = max(new_w - scl_new_w, 0)
    d_h = max(new_h - scl_new_h, 0)
    # center image with padding on top/bottom or left/right
    top, bottom = d_h // 2, d_h - (d_h // 2)
    left, right = d_w // 2, d_w - (d_w // 2)
    pad_rsz_img = cv2.copyMakeBorder(rsz_img, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT,
                                     value=color)
    return pad_rsz_img


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, np.ndarray):
        boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
        boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
        boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
        boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2
    else:  # torch.Tensor
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
