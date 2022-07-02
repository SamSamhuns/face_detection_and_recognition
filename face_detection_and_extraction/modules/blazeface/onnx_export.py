import time

import cv2
import onnx
import torch
import numpy as np
from modules.blazeface.blazeface import BlazeFace, plot_detections


def preprocess_onnx(cv2_img, back_model=True):
    if back_model:
        w, h = 256, 256
    else:
        w, h = 128, 128
    # BGR to RGB and resize to network input size
    img = cv2.resize(cv2_img[..., ::-1], (w, h))
    # change order to Channel, Height, Width and add batch dimension
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    # normalize to range [-1,1]
    img = img.astype(np.float32) / 127.5 - 1.0
    return img


def export_onnx(img_path, onnx_savepath="weights/blazeface/blazefaceback.onnx", dynamic=True, simplify=True):
    net = BlazeFace(back_model=True)
    net.load_weights("weights/blazeface/blazefaceback.pth")
    net.load_anchors("weights/blazeface/anchorsback.npy")

    orig_img = cv2.imread(img_path)
    onnx_input = preprocess_onnx(orig_img, back_model=True)

    print("onnx input shape", onnx_input.shape)
    output = net(onnx_input)
    print("onnx output shape", output[0].shape, output[1].shape)
    torch.onnx.export(net, onnx_input, onnx_savepath, verbose=False, opset_version=12,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      input_names=['images'],
                      output_names=['output0', 'output1'],
                      dynamic_axes={'images': {0: 'batch'},
                                    'output0': {0: 'batch'},
                                    'output1': {0: 'batch'}
                                    } if dynamic else None)

    # Checks
    model_onnx = onnx.load(onnx_savepath)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    print("ONNX model input shape")
    print(model_onnx.graph.input)
    print("ONNX model output shape")
    print(model_onnx.graph.output)

    # Simplify
    if simplify:
        try:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=dynamic,
                input_shapes={'images': list(onnx_input.shape)} if dynamic else None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, onnx_savepath)
        except Exception as e:
            print(f'simplifier failure: {e}')


def inference_onnx(img_path, onnx_savepath="weights/blazeface/blazefaceback.onnx"):
    import onnxruntime
    ort_session = onnxruntime.InferenceSession(onnx_savepath, None)

    use_numpy = True
    dummy_model = BlazeFace(back_model=True)
    dummy_model.load_weights("weights/blazeface/blazefaceback.pth")
    dummy_model.load_anchors("weights/blazeface/anchorsback.npy", use_numpy=use_numpy)

    orig_img = cv2.imread("test.jpeg")
    onnx_input = preprocess_onnx(orig_img, back_model=True)
    start_time = time.time()

    outputs = ort_session.run(None, {
        "images": onnx_input
    })
    print("ONNX inference output shapes:", outputs[0].shape, outputs[1].shape)

    # Postprocess the raw predictions:
    detections = dummy_model._tensors_to_detections(
        outputs[0], outputs[1], dummy_model.anchors, use_numpy=use_numpy)

    # Non-maximum suppression to remove overlapping detections:
    filtered_detections = []
    for detection in detections:
        faces = dummy_model._weighted_non_max_suppression(
            detection, use_numpy=use_numpy)
        if use_numpy:
            faces = np.stack(faces) if len(faces) > 0 \
                else np.zeros((0, 17))
        else:
            faces = torch.stack(faces) if len(faces) > 0 \
                else torch.zeros((0, 17))
        filtered_detections.append(faces)

    back_detections = filtered_detections[0]
    inference_time = time.time() - start_time
    print(
        f"ONNX single img inf time:{inference_time:.2f}s, FPS: {1/inference_time}")
    output_path = "onnx_output.jpg"
    plot_detections(orig_img, back_detections,
                    with_keypoints=True, output_path=output_path)
    print(f"Saving output in {output_path}")


def main():
    export_onnx("test.jpeg")
    inference_onnx("test.jpeg")


if __name__ == "__main__":
    main()
