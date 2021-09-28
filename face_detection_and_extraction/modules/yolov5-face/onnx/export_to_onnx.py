import argparse
import time

import onnx
import torch
import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='weights/yolov5n/yolov5n-face.pt',
                        help='pytorch weight path. (default: %(default)s)')
    parser.add_argument('--img_size', nargs='+', type=int,
                        default=[640, 640], help='image size (h, w) (default: %(default)s)')
    parser.add_argument('--batch_size', type=int,
                        default=1, help='batch size. (default: %(default)s)')
    parser.add_argument('--simplify', action='store_true',
                        help='simplify onnx graph')
    parser.add_argument('--dynamic', action='store_true',
                        help='dynamic input output shapes')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    model = attempt_load(
        opt.weights, map_location=torch.device('cpu'))  # load FP32 model
    model.eval()
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    # verify img_size are gs-multiples
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]

    # Input
    # image size(1,3,320,192) iDetection
    img = torch.zeros(opt.batch_size, 3, *opt.img_size)

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        if isinstance(m, models.common.ShuffleV2Block):  # shufflenet block nn.SiLU
            for i in range(len(m.branch1)):
                if isinstance(m.branch1[i], nn.SiLU):
                    m.branch1[i] = SiLU()
            for i in range(len(m.branch2)):
                if isinstance(m.branch2[i], nn.SiLU):
                    m.branch2[i] = SiLU()
    model.model[-1].export = True  # set Detect() layer export=True
    y = model(img)  # dry run
    print("Dry run results", len(y), y[0].shape)
    # ONNX export
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    f = opt.weights.replace('.pt', '.onnx')  # filename
    model.fuse()  # only for ONNX
    print(f"ONNX input shape {img.shape}")
    output_names = ['stride_' + str(int(x)) for x in model.stride]
    in_shape_dict = {'images': {0: 'batch', 2: 'height', 3: 'width'}}
    out_shape_dict = {lname: {0: "batch", 1: "channel", 2: "height", 3: "width", 4: "stride"}
                      for lname in output_names}
    io_shapes_dict = {**in_shape_dict, **out_shape_dict}
    torch.onnx.export(model, img, f, verbose=False,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      opset_version=11,
                      input_names=['images'],
                      output_names=output_names,
                      dynamic_axes={**io_shapes_dict}
                      if opt.dynamic else None)

    # checks
    onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)
    print('ONNX export success, saved as %s' % f)
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' %
          (time.time() - t))
    print(onnx_model.graph.input)
    print(onnx_model.graph.output)

    # simplify
    if opt.simplify:
        try:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}...')
            onnx_model, check = onnxsim.simplify(
                onnx_model,
                dynamic_input_shape=opt.dynamic,
                input_shapes={'images': list(img.shape)} if opt.dynamic else None)
            assert check, 'assert check failed'
            onnx.save(onnx_model, f)
        except Exception as e:
            print(f'simplifier failure: {e}')
