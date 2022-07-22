import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np
import cv2
import time

from modules.utils.image import scale_coords


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BlazeBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)


class FinalBlazeBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(FinalBlazeBlock, self).__init__()
        # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=kernel_size, stride=2, padding=0,
                      groups=channels, bias=True),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = F.pad(x, (0, 2, 0, 2), "constant", 0)

        return self.act(self.convs(h))


class BlazeFace(nn.Module):
    """The BlazeFace face detection model from MediaPipe.

    The version from MediaPipe is simpler than the one in the paper;
    it does not use the "double" BlazeBlocks.
    Because we won't be training this model, it doesn't need to have
    batchnorm layers. These have already been "folded" into the conv
    weights by TFLite.
    The conversion to PyTorch is fairly straightforward, but there are
    some small differences between TFLite and PyTorch in how they handle
    padding on conv layers with stride 2.
    This version works on batches, while the MediaPipe version can only
    handle a single image at a time.
    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/google/mediapipe/
    """

    def __init__(self, back_model=False):
        super(BlazeFace, self).__init__()

        # These are the settings from the MediaPipe example graphs
        # mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt
        # and mediapipe/graphs/face_detection/face_detection_back_mobile_gpu.pbtxt
        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 16
        self.score_clipping_thresh = 100.0
        self.back_model = back_model
        if back_model:
            self.x_scale = 256.0
            self.y_scale = 256.0
            self.h_scale = 256.0
            self.w_scale = 256.0
            self.min_score_thresh = 0.65
        else:
            self.x_scale = 128.0
            self.y_scale = 128.0
            self.h_scale = 128.0
            self.w_scale = 128.0
            self.min_score_thresh = 0.75
        self.min_suppression_threshold = 0.3

        self._define_layers()

    def _define_layers(self):
        if self.back_model:
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=24,
                          kernel_size=5, stride=2, padding=0, bias=True),
                nn.ReLU(inplace=True),

                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24, stride=2),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 48, stride=2),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 96, stride=2),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
            )
            self.final = FinalBlazeBlock(96)
            self.classifier_8 = nn.Conv2d(96, 2, 1, bias=True)
            self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)

            self.regressor_8 = nn.Conv2d(96, 32, 1, bias=True)
            self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)
        else:
            self.backbone1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=24,
                          kernel_size=5, stride=2, padding=0, bias=True),
                nn.ReLU(inplace=True),

                BlazeBlock(24, 24),
                BlazeBlock(24, 28),
                BlazeBlock(28, 32, stride=2),
                BlazeBlock(32, 36),
                BlazeBlock(36, 42),
                BlazeBlock(42, 48, stride=2),
                BlazeBlock(48, 56),
                BlazeBlock(56, 64),
                BlazeBlock(64, 72),
                BlazeBlock(72, 80),
                BlazeBlock(80, 88),
            )

            self.backbone2 = nn.Sequential(
                BlazeBlock(88, 96, stride=2),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
            )
            self.classifier_8 = nn.Conv2d(88, 2, 1, bias=True)
            self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)

            self.regressor_8 = nn.Conv2d(88, 32, 1, bias=True)
            self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)

    def forward(self, x):
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        x = F.pad(x, (1, 2, 1, 2), "constant", 0)

        b = x.shape[0]      # batch size, needed for reshaping later

        if self.back_model:
            x = self.backbone(x)           # (b, 16, 16, 96)
            h = self.final(x)              # (b, 8, 8, 96)
        else:
            x = self.backbone1(x)           # (b, 88, 16, 16)
            h = self.backbone2(x)           # (b, 96, 8, 8)

        # Note: Because PyTorch is NCHW but TFLite is NHWC, we need to
        # permute the output from the conv layers before reshaping it.

        c1 = self.classifier_8(x)       # (b, 2, 16, 16)
        c1 = c1.permute(0, 2, 3, 1)     # (b, 16, 16, 2)
        c1 = c1.reshape(b, -1, 1)       # (b, 512, 1)

        c2 = self.classifier_16(h)      # (b, 6, 8, 8)
        c2 = c2.permute(0, 2, 3, 1)     # (b, 8, 8, 6)
        c2 = c2.reshape(b, -1, 1)       # (b, 384, 1)

        c = torch.cat((c1, c2), dim=1)  # (b, 896, 1)

        r1 = self.regressor_8(x)        # (b, 32, 16, 16)
        r1 = r1.permute(0, 2, 3, 1)     # (b, 16, 16, 32)
        r1 = r1.reshape(b, -1, 16)      # (b, 512, 16)

        r2 = self.regressor_16(h)       # (b, 96, 8, 8)
        r2 = r2.permute(0, 2, 3, 1)     # (b, 8, 8, 96)
        r2 = r2.reshape(b, -1, 16)      # (b, 384, 16)

        r = torch.cat((r1, r2), dim=1)  # (b, 896, 16)
        return [r, c]

    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return self.classifier_8.weight.device

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_anchors(self, path, use_numpy=False):
        if use_numpy:
            self.anchors = np.load(path).astype(np.float32)
        else:
            self.anchors = torch.tensor(
                np.load(path), dtype=torch.float32, device=self._device())
            assert(self.anchors.ndimension() == 2)
            assert(self.anchors.shape[0] == self.num_anchors)
            assert(self.anchors.shape[1] == 4)

    def _preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        return x.float() / 127.5 - 1.0

    def predict_on_image(self, img):
        """Makes a prediction on a single image.
        Arguments:
            img: a NumPy array of shape (H, W, 3) or a PyTorch tensor of
                 shape (3, H, W). The image's height and width should be
                 128 pixels.
        Returns:
            A tensor with face detections.
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.copy()).permute((2, 0, 1))

        return self.predict_on_batch(img.unsqueeze(0))[0]

    def predict_on_batch(self, x, use_numpy_for_post_proc=False):
        """Makes a prediction on a batch of images.
        Arguments:
            x: a NumPy array of shape (b, H, W, 3) or a PyTorch tensor of
               shape (b, 3, H, W). The height and width should be 128 pixels.
            use_numpy_for_post_proc: bool = Use only numpy functions for postprocessing
                Inference still requires pytorch
        Returns:
            A list containing a tensor of face detections for each image in
            the batch. If no faces are found for an image, returns a tensor
            of shape (0, 17).
        Each face detection is a PyTorch tensor consisting of 17 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 6 keypoints
            - confidence score
        """
        use_numpy = use_numpy_for_post_proc
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))

        assert x.shape[1] == 3
        if self.back_model:
            assert x.shape[2] == 256
            assert x.shape[3] == 256
        else:
            assert x.shape[2] == 128
            assert x.shape[3] == 128

        # 1. Preprocess the images into tensors:
        x = x.to(self._device())
        x = self._preprocess(x)

        # 2. Run the neural network:
        with torch.no_grad():
            out = self.__call__(x)

        # 3. Postprocess the raw predictions:
        detections = self._tensors_to_detections(
            out[0], out[1], self.anchors, use_numpy=use_numpy)

        # 4. Non-maximum suppression to remove overlapping detections:
        filtered_detections = []
        for i in range(len(detections)):
            faces = self._weighted_non_max_suppression(
                detections[i], use_numpy=use_numpy)
            if use_numpy:
                faces = np.stack(faces) if len(
                    faces) > 0 else np.zeros((0, 17))
            else:
                faces = torch.stack(faces) if len(
                    faces) > 0 else torch.zeros((0, 17))
            filtered_detections.append(faces)

        return filtered_detections

    def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors, use_numpy=False):
        """The output of the neural network is a tensor of shape (b, 896, 16)
        containing the bounding box regressor predictions, as well as a tensor
        of shape (b, 896, 1) with the classification confidences.
        This function converts these two "raw" tensors into proper detections.
        Returns a list of (num_detections, 17) tensors, one for each image in
        the batch.
        This is based on the source code from:
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
        """
        assert raw_box_tensor.ndim == 3
        assert raw_box_tensor.shape[1] == self.num_anchors
        assert raw_box_tensor.shape[2] == self.num_coords

        assert raw_score_tensor.ndim == 3
        assert raw_score_tensor.shape[1] == self.num_anchors
        assert raw_score_tensor.shape[2] == self.num_classes

        assert raw_box_tensor.shape[0] == raw_score_tensor.shape[0]

        detection_boxes = self._decode_boxes(
            raw_box_tensor, anchors, use_numpy=use_numpy)

        thresh = self.score_clipping_thresh
        if use_numpy:
            raw_score_tensor = np.clip(raw_score_tensor, -thresh, thresh)
            detection_scores = np.squeeze(
                (lambda x: 1 / (1 + np.exp(-x)))(raw_score_tensor), axis=-1)
        else:
            raw_score_tensor = raw_score_tensor.clamp(-thresh, thresh)
            detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)

        # Note: we stripped off the last dimension from the scores tensor
        # because there is only has one class. Now we can simply use a mask
        # to filter out the boxes with too low confidence.
        mask = detection_scores >= self.min_score_thresh

        # Because each image from the batch can have a different number of
        # detections, process them one at a time using a loop.
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            if use_numpy:
                scores = np.expand_dims(detection_scores[i, mask[i]], axis=-1)
                output_detections.append(
                    np.concatenate((boxes, scores), axis=-1))
            else:
                scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
                output_detections.append(torch.cat((boxes, scores), dim=-1))
        return output_detections

    def _decode_boxes(self, raw_boxes, anchors, use_numpy=False):
        """Converts the predictions into actual coordinates using
        the anchor boxes. Processes the entire batch at once.
        """
        boxes = np.zeros(raw_boxes.shape) if use_numpy else \
            torch.zeros_like(raw_boxes)

        x_center = raw_boxes[..., 0] / self.x_scale * \
            anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale * \
            anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / self.w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * anchors[:, 3]

        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax

        for k in range(6):
            offset = 4 + k * 2
            keypoint_x = raw_boxes[..., offset] / \
                self.x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / \
                self.y_scale * anchors[:, 3] + anchors[:, 1]
            boxes[..., offset] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    def _weighted_non_max_suppression(self, detections, use_numpy=False):
        """The alternative NMS method as mentioned in the BlazeFace paper:
        "We replace the suppression algorithm with a blending strategy that
        estimates the regression parameters of a bounding box as a weighted
        mean between the overlapping predictions."
        The original MediaPipe code assigns the score of the most confident
        detection to the weighted detection, but we take the average score
        of the overlapping detections.
        The input detections should be a Tensor of shape (count, 17).
        Returns a list of PyTorch tensors, one for each detected face.

        This is based on the source code from:
        mediapipe/calculators/util/non_max_suppression_calculator.cc
        mediapipe/calculators/util/non_max_suppression_calculator.proto
        """
        if len(detections) == 0:
            return []
        output_detections = []
        # Sort the detections from highest to lowest score.
        if use_numpy:
            remaining = np.argsort(-1 * detections[:, 16])
        else:
            remaining = torch.argsort(detections[:, 16], descending=True)

        while len(remaining) > 0:
            detection = detections[remaining[0]]

            # Compute the overlap between the first box and the other
            # remaining boxes. (Note that the other_boxes also include
            # the first_box.)
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = overlap_similarity(
                first_box, other_boxes, use_numpy=use_numpy)

            # If two detections don't overlap enough, they are considered
            # to be from different faces.
            mask = ious > self.min_suppression_threshold
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            # Take an average of the coordinates from the overlapping
            # detections, weighted by their confidence scores.
            weighted_detection = detection.copy() if use_numpy else detection.clone()
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :16]
                scores = detections[overlapping, 16:17]
                total_score = scores.sum()
                weighted = (coordinates * scores).sum(0) / total_score
                weighted_detection[:16] = weighted
                weighted_detection[16] = total_score / len(overlapping)

            output_detections.append(weighted_detection)

        return output_detections


def plot_detections(cv2_img, detections, model_in_HW, threshold=0.5, with_keypoints=True):
    H, W = model_in_HW
    h, w = cv2_img.shape[:2]
    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)
    # filter weak detections
    detections = detections[detections[:, -1] > threshold]
    # change (ymin, xmin, ymax, xmax) tp (xmin, ymin, xmax, ymax) format
    detections = detections[:, np.array(
        [1, 0, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])]
    # rescale detections to orig image size taking the padding into account
    boxes = detections * \
        np.array([W, H, W, H, W, H, W, H, W, H, W, H, W, H, W, H])
    boxes = scale_coords((H, W), boxes, (h, w)).round()

    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box[:4].astype('int')
        cv2.rectangle(cv2_img, (xmin, ymin), (xmax, ymax),
                      color=(0, 0, 255), thickness=1)
        if with_keypoints:
            for k in range(6):
                kp_x = int(boxes[i, 4 + k * 2])
                kp_y = int(boxes[i, 4 + k * 2 + 1])
                cv2.circle(cv2_img, (kp_x, kp_y), radius=1,
                           color=(255, 0, 0), thickness=1)
    # print("Found %d faces" % boxes.shape[0])


# IOU code from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py

def intersect(box_a, box_b, use_numpy=False):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    if use_numpy:
        A = box_a.shape[0]
        B = box_b.shape[0]
        max_xy = np.minimum(np.broadcast_to(np.expand_dims(box_a[:, 2:], 1), (A, B, 2)),
                            np.broadcast_to(np.expand_dims(box_b[:, 2:], 0), (A, B, 2)))
        min_xy = np.maximum(np.broadcast_to(np.expand_dims(box_a[:, :2], 1), (A, B, 2)),
                            np.broadcast_to(np.expand_dims(box_b[:, :2], 0), (A, B, 2)))
        inter = np.clip((max_xy - min_xy), 0, None)
        return inter[:, :, 0] * inter[:, :, 1]
    else:
        A = box_a.size(0)
        B = box_b.size(0)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b, use_numpy=False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b, use_numpy)
    if use_numpy:
        area_a = np.broadcast_to(np.expand_dims(
            ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])), axis=1), inter.shape)
        area_b = np.broadcast_to(np.expand_dims(
            ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])), axis=0), inter.shape)
    else:
        area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
                  ).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
                  ).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def overlap_similarity(box, other_boxes, use_numpy=False):
    """Computes the IOU between a bounding box and set of other boxes."""
    if use_numpy:
        return jaccard(np.expand_dims(box, axis=0), other_boxes, use_numpy).squeeze(0)
    else:
        return jaccard(box.unsqueeze(0), other_boxes, use_numpy).squeeze(0)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    back_net = BlazeFace(back_model=True).to(device)
    back_net.load_weights("weights/blazeface/blazefaceback.pth")
    back_net.load_anchors("weights/blazeface/anchorsback.npy")

    orig_img = cv2.imread("modules/blazeface/test.jpeg")
    img = cv2.resize(orig_img[..., ::-1], (256, 256))
    start_time = time.time()

    back_detections = back_net.predict_on_image(img)
    inference_time = time.time() - start_time
    print(f"Single img inf time:{inference_time:.2f}s, FPS:{1/inference_time}")
    plot_detections(orig_img, back_detections, with_keypoints=True)


if __name__ == "__main__":
    main()
