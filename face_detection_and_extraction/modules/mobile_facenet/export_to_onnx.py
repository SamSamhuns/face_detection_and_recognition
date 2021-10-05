import onnx
import torch
import onnxruntime
import numpy as np
from mobile_facenet import MobileFaceNet


def export_to_onnx(pt_weight_path="weights/mobile_facenet/MobileFace_Net",
                   onnx_save_path="weights/mobile_facenet/mobile_facenet.onnx",
                   dynamic=True,
                   simplify=True,
                   verbose=True,
                   embedding_vectors=512,
                   input_size=(112, 112)):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # embeding size is 512 (feature vector)
    model = MobileFaceNet(embedding_vectors).to(device)
    model.load_state_dict(torch.load(
        pt_weight_path, map_location=lambda storage, loc: storage))
    model.eval()
    if verbose:
        print('mobile_facenet face embedding extraction model loaded')

    w, h = input_size
    dummy_input = torch.Tensor(np.ones([2, 3, h, w]))
    torch_out = model(dummy_input)
    if verbose:
        print(f"torch input shape: {dummy_input.shape}")
        print(f"torch output shape: {torch_out.shape}")

    in_name, out_name = 'images', 'embedding'
    in_shape_dict = {in_name:
                     {0: 'batch'}}
    out_shape_dict = {out_name:
                      {0: "batch"}}
    io_shapes_dict = {**in_shape_dict, **out_shape_dict}

    torch.onnx.export(model, dummy_input, onnx_save_path, verbose=False,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      opset_version=12,
                      input_names=[in_name],
                      output_names=[out_name],
                      dynamic_axes={**io_shapes_dict}
                      if dynamic else None)

    onnx_model = onnx.load(onnx_save_path)
    if verbose:
        print("ONNX model input graph")
        print(onnx_model.graph.input)
        print("ONNX model output graph")
        print(onnx_model.graph.output)

    # simplify
    if simplify:
        try:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}...')
            onnx_model, check = onnxsim.simplify(
                onnx_model,
                dynamic_input_shape=dynamic,
                input_shapes={'images': list(dummy_input.shape)} if dynamic else None)
            assert check, 'assert check failed'
            onnx.save(onnx_model, onnx_save_path)
            print(
                f"ONNX model simplification complete. Model saved to {onnx_save_path}")
        except Exception as e:
            print(f'simplifier failure: {e}')
    else:
        print(f"ONNX export complete. Model saved to {onnx_save_path}")

    text_onnx_inference(onnx_save_path, dummy_input, torch_out)


def text_onnx_inference(onnx_save_path, onnx_input, torch_out):

    ort_session = onnxruntime.InferenceSession(onnx_save_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(onnx_input)}
    ort_out = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(
        to_numpy(torch_out), ort_out[0], rtol=1e-03, atol=1e-05)
    print("\x1b[6;30;42m SUCCESS: \x1b[0m Exported model has been tested with ONNXRuntime, and the results are similar")


def main():
    export_to_onnx()


if __name__ == "__main__":
    main()
