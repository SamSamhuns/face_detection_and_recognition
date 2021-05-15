# import the necessary packages
import numpy as np
import argparse
import cv2
import os


def main():
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image",
                        help="path to input image")
    parser.add_argument("-v", "--video",
                        help="path to input video")
    parser.add_argument("-w", "--webcam",
                        action='store_true',
                        default=False,
                        help="webcam mode")
    parser.add_argument("-p", "--prototxt",
                        default="./dnn_caffe/deploy.prototxt.txt",
                        help="path to 'deploy' prototxt file")
    parser.add_argument("-m", "--model",
                        default="./dnn_caffe/res10_300x300_ssd_iter_140000.caffemodel",
                        help="path to pre-trained model")
    parser.add_argument("-t", "--threshold",
                        type=float,
                        default=0.5,
                        help="minimum probability to filter weak detections")
    args = parser.parse_args()

    # load model
    net = load_net(args.model, args.prototxt)

    # choose inference mode
    if args.webcam and args.image is None and args.video is None:
        inference_webcam(net, args.threshold)
    elif args.image and args.video is None and args.webcam is False:
        inference_img(net, args.image, args.threshold)
    elif args.video and args.image is None and args.webcam is False:
        inference_vid(net, args.video, args.threshold)
    else:
        print("Only one mode is allowed")
        print("\tpython dnn_detect_face -w              # webcam mode")
        print("\tpython dnn_detect_face -i img_path     # image mode")
        print("\tpython dnn_detect_face -v vid_path     # video mode")


def load_net(model, prototxt):
    # load face detection model
    fname, fext = os.path.splitext(model)
    if fext == ".caffemodel":
        net = cv2.dnn.readNetFromCaffe(prototxt, model)
    elif fext == ".pb":
        net = cv2.dnn.readNetFromTensorflow(model, prototxt)
    else:
        raise NotImplementedError(
            f"[ERROR] model with extension {fext} not implemented")
    return net


def inference_model(net, cv2_img):
    blob = cv2.dnn.blobFromImage(cv2.resize(
        cv2_img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    return faces


def inference_img(net, img, threshold, waitKey_val=0):
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    if isinstance(img, str):
        if os.path.exists(img):
            image = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        image = img
    else:
        raise Exception("image cannot be read")

    (h, w) = image.shape[:2]
    # pass the blob through the network and
    # obtain the detections and predictions
    detections = inference_model(net, image)
    count = 0

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability)
        # associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter weak detections
        if confidence > threshold:
            count += 1
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY),
                          (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 0, 255), 2)
    # print(f"Num of faces detected = {count} ")
    # show the output image
    cv2.imshow("output", image)
    cv2.waitKey(waitKey_val)


def inference_vid(net, vid, threshold):
    cap = cv2.VideoCapture(vid)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        inference_img(net, frame, threshold, waitKey_val=5)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def inference_webcam(net, threshold):
    inference_vid(net, 0, threshold)


def batch_inference_img(net, cv2_img_nparray):
    """reference func for batched DNN inference
    """
    image = cv2_img_nparray
    blob = cv2.dnn.blobFromImages(
        image, 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    count = 0
    threshold = 0.5
    img_idx = 0
    (h, w) = image[0].shape[:2]

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability)
        # associated with the prediction
        confidence = detections[0, 0, i, 2]
        det_img_idx = int(detections[0, 0, i, 0])

        # filter weak detections
        if confidence > threshold:
            count += 1
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image[det_img_idx], (startX, startY),
                          (endX, endY), (0, 0, 255), 2)
            cv2.putText(image[det_img_idx], text, (startX, y),
                        cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 0, 255), 2)
        if (i + 1) % 200 == 0:
            cv2.imshow("output", image[img_idx])
            cv2.waitKey(0)
            img_idx += 1


if __name__ == "__main__":
    main()
