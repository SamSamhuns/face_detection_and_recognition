from mtcnn.mtcnn import MTCNN
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
    parser.add_argument("-t", "--threshold",
                        type=float,
                        default=0.5,
                        help="minimum probability to filter weak detections")
    args = parser.parse_args()

    # load model
    net = MTCNN()

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


# draw an image with detected objects
def draw_image_with_boxes(img, result_list, threshold, waitKey_val=0):
    # plot each box
    for result in result_list:
        conf = result['confidence']
        if conf > threshold:
            # get coordinates
            x, y, width, height = result['box']
            startX, startY, endX, endY = x, y, x + width, y + height
            cv2.rectangle(img, (startX, startY),
                          (endX, endY), (0, 0, 255), 2)

            # draw confidence on faces
            text = "{:.2f}%".format(conf * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(img, text, (startX, y),
                        cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 0, 255), 2)

            # draw the keypoint dots
            for key, value in result['keypoints'].items():
                cv2.circle(img, center=value, radius=2, color=(0, 0, 255))
    cv2.imshow("output", img)
    cv2.waitKey(waitKey_val)


def inference_img(net, img, threshold, waitKey_val=0):
    """ run inference on img with mtcnn model
    """
    if isinstance(img, str):
        if os.path.exists(img):
            image = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        image = img
    else:
        raise Exception("image/frame cannot be read")

    # detect faces in the image
    faces = net.detect_faces(image)
    # display faces on the original image
    draw_image_with_boxes(image, faces, threshold, waitKey_val)


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


if __name__ == "__main__":
    main()
