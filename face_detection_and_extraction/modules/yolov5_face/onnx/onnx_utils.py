import time

import torch
import torchvision
import numpy as np
from modules.common_utils import check_img_size, pad_resize_image, scale_coords


def preprocess_image(cv2_image, input_size=(640, 640)):
    """preprocesses a cv2_image BGR
    args:
        cv2_image = cv2 image
        in_size: in_width, in_height
    """
    cv2_image = cv2_image[..., ::-1]  # BGR2RGB
    # make sure img dims are divisible by model stride
    in_w, in_h = tuple(map(check_img_size, input_size))
    pad_resized = pad_resize_image(cv2_image, (in_w, in_h))
    img = np.transpose(pad_resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    return img


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


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # (pixels) maximum and minimum box width and height
    max_wh = 4096  # min_wh = 2
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)
              ] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lxi = labels[xi]
            v = torch.zeros((len(lxi), nc + 5), device=x.device)
            v[:, :4] = lxi[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lxi)), lxi[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float(
            ) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def get_bboxes_confs_areas(dets, det_thres, bbox_area_thres, orig_size, in_size):
    """
    Returns a tuple of bounding boxes, bbox confidence scores, and bbox area percentages
    args:
        dets: np.ndarray of fmt [xmin, ymin, xmax, ymax, conf]
        det_thres: float bbox detection threshold
        bbox_area_thres: float bounding box area min threshold
        orig_size: original image size (width, height)
        in_size: model input size (width, height)
    """
    w, h = orig_size
    iw, ih = in_size
    if not isinstance(dets, np.ndarray):
        dets = dets.numpy()
    # filter detections below threshold
    dets = dets[dets[..., 4] > det_thres]
    # only select bboxes with area greater than 0.15% of total area of frame
    total_area = iw * ih
    bbox_area = ((dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1]))
    bbox_area_perc = (100 * bbox_area / total_area)
    dets = dets[bbox_area_perc > bbox_area_thres]

    bbox_confs = dets[..., 4]
    boxes = dets[..., :4]
    # rescale detections to orig image size taking the padding into account
    boxes = scale_coords((ih, iw), boxes, (h, w)).round()

    return boxes, bbox_confs, bbox_area_perc


def inference_onnx_model_yolov5_face(net, cv2_img, input_size, official=False, **kwargs):
    # note any other kwargs are ignored
    resized = preprocess_image(cv2_img, input_size=input_size)
    outputs = net.run(None, {"images": resized})

    if official:  # for official yolov5 models
        detections = torch.from_numpy(np.array(outputs[0]))
        detections = non_max_suppression(
            detections, conf_thres=0.4, iou_thres=0.5, agnostic=False)
    else:         # for yolov5-face models
        outputx = conv_strides_to_anchors(outputs, "cpu")
        detections = w_non_max_suppression(
            outputx, num_classes=1, conf_thres=0.4, nms_thres=0.3)
    return detections[0]
