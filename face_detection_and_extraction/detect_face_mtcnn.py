from mtcnn.mtcnn import MTCNN
import numpy as np
import cv2
import os

from modules.utils.parser import get_argparse
from modules.utils.files import get_file_type


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
            for value in result['keypoints'].values():
                cv2.circle(img, center=value, radius=2, color=(0, 0, 255))
    cv2.imshow("output", img)
    cv2.waitKey(waitKey_val)


def inference_img(net, img, threshold, waitKey_val=0):
    """ run inference on img with mtcnn model
    """
    if isinstance(img, str):
        if os.path.exists(img):
            image = cv2.imread(img)
        else:
            raise FileNotFoundError(f"{img} does not exist")
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
    ret, frame = cap.read()

    while ret:
        # inference and display the resulting frame
        inference_img(net, frame, threshold, waitKey_val=5)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()


def inference_webcam(net, cam_index, threshold):
    inference_vid(net, cam_index, threshold)


def main():
    parser = get_argparse(description="MTCNN face detection")
    parser.remove_arguments(["model", "bbox_area_thres"])
    args = parser.parse_args()

    # load model
    net = MTCNN()

    # choose inference mode
    input_type = get_file_type(args.input_src)
    if input_type == "camera":
        inference_webcam(net, int(args.input_src), args.det_thres)
    elif input_type == "video":
        inference_vid(net, args.input_src, args.det_thres)
    elif input_type == "image":
        inference_img(net, args.input_src, args.det_thres)
    else:
        print("File type or inference mode not recognized. Use --help")


if __name__ == "__main__":
    main()
