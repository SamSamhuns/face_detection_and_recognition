import cv2
import numpy as np


def inference_onnx_model_mobile_facenet(feature_net, face, face_feat_in_size=(112, 112)):
    """
    Get face features with mobile_facenet
    args:
        feature_net: ort session mobile_facenet model
        face: cv2 images of face
        face_feat_in_size: (W, H) input size of network
    """
    face = (cv2.resize(face, face_feat_in_size) - 127.5) / 127.5
    # HWC to BCHW
    face = np.expand_dims(np.transpose(face, (2, 0, 1)),
                          axis=0).astype(np.float32)
    features = feature_net.run(None, {"images": face})  # BGR fmt
    return features[0][0]
