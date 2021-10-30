import os
import time
import argparse

import onnx
import torch
import onnxsim
import numpy as np
import onnxruntime

from utils import read_json
from model.model import ResMLP


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


def onnx_export(args):
    onnx_save_dir = os.path.dirname(args.onnx_save_path)
    os.makedirs(onnx_save_dir, exist_ok=True)
    t = time.time()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    # load model from pt chkpt and json config file
    model = ResMLP(**read_json(args.pt_model_config_path)['arch']['args'])
    checkpoint = args.pt_model_path
    checkpoint = torch.load(checkpoint, map_location=torch.device(device))
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    in_shape = list(map(int, args.input_shape))
    dummy_input = torch.ones(in_shape).to(device)
    torch_out = model(dummy_input)  # dry run
    if args.verbose:
        print("Pytorch test run:")
        print(f"\ttorch input shape: {dummy_input.shape}")
        print(f"\ttorch output shape: {torch_out.shape}")

    # ONNX export
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    print(f"\tONNX input shape {dummy_input.shape}")
    input_name = 'images'
    output_name = 'output'
    in_shape_dict = {input_name: {0: 'batch'}}
    out_shape_dict = {output_name: {0: "batch"}}
    io_shapes_dict = {**in_shape_dict, **out_shape_dict}
    torch.onnx.export(model, dummy_input, args.onnx_save_path, verbose=False,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      opset_version=11,
                      input_names=[input_name],
                      output_names=[output_name],
                      dynamic_axes={**io_shapes_dict}
                      if args.dynamic else None)

    # checks
    onnx_model = onnx.load(args.onnx_save_path)
    onnx.checker.check_model(onnx_model)
    print(f'ONNX export success, saved at {args.onnx_save_path}')
    print(f'\nExport done ({time.time() - t:.2f}) (%.2fs).' +
          '\nVisualize with https://github.com/lutzroeder/netron.')
    if args.verbose:
        print(onnx_model.graph.input)
        print(onnx_model.graph.output)

    # simplify
    if args.simplify:
        try:
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}...')
            onnx_model, check = onnxsim.simplify(
                onnx_model,
                dynamic_input_shape=args.dynamic,
                input_shapes={'images': list(dummy_input.shape)} if args.dynamic else None)
            assert check, 'assert check failed'
            onnx.save(onnx_model, args.onnx_save_path)
            print("ONNX model simplified")
        except Exception as e:
            print(f'simplifier failure: {e}')

    text_onnx_inference(args.onnx_save_path, dummy_input, torch_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--onnx_save_path', type=str,
                        default='saved/age.onnx',
                        help='ONNX model save path. (default: %(default)s)')
    parser.add_argument('-pt', '--pt_model_path', type=str, required=True,
                        help='Path to pytorch .pt/pth model checkpoint file')
    parser.add_argument('-c', '--pt_model_config_path', type=str, required=True,
                        help='Path to pytorch model json config path')
    parser.add_argument('-is', '--input_shape', nargs='+', type=int,
                        default=[2, 512], help='Input data/layer shape (default: %(default)s)')
    parser.add_argument('--simplify', action='store_true',
                        help='simplify onnx graph')
    parser.add_argument('-d', '--dynamic', action='store_true',
                        help='dynamic input output shapes')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose mode')
    args = parser.parse_args()
    print(args)
    onnx_export(args)
