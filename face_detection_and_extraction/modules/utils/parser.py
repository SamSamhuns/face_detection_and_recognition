import argparse
from typing import List


class ArgumentParserMod(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def remove_argument(self, arg: str):
        """
        Remove argument from argparse object
        args:
            arg: argument name without leading dashes
        """
        for action in self._actions:
            if (vars(action)['option_strings'] and vars(action)['option_strings'][0] == arg) \
                    or vars(action)['dest'] == arg:
                self._remove_action(action)

        for action in self._action_groups:
            vars_action = vars(action)
            var_group_actions = vars_action['_group_actions']
            for x in var_group_actions:
                if x.dest == arg:
                    var_group_actions.remove(x)
                    return

    def remove_arguments(self, arg_list: List[str]):
        """
        Remove list of arguments from argparse object
        args:
        """
        _ = [self.remove_argument(arg) for arg in arg_list]


def get_argparse(*args, **kwargs):
    """
    get base argparse arguments
        remove arguments with parser.remove_argparse_option(...)
        add new arguments with parser.add_argument(...)
    """
    parser = ArgumentParserMod(*args, **kwargs)
    parser.add_argument("-i", "--input_src", default='0', dest="input_src",
                        help=("Path to input image/video/cam_index:\n"
                              "\t IMAGE_MODE       -i <PATH_TO_IMG>\n"
                              "\t VIDEO_MODE       -i <PATH_TO_VID>\n"
                              "\t CAM MODE:Default -i <CAM_INDEX>  -i 0 (for webcam)\n"))
    parser.add_argument("--md", "--model", dest="model",
                        default="weights/face_detection_caffe/res10_300x300_ssd_iter_140000.caffemodel",
                        help='Path to model file. (default: %(default)s)')
    parser.add_argument("--dt", "--det_thres", dest="det_thres",
                        type=float, default=0.70,
                        help='score to filter weak detections. (default: %(default)s)')
    parser.add_argument("--at", "--bbox_area_thres", dest="bbox_area_thres",
                        type=float, default=0.12,
                        help='bbox_area * 100/image_area perc thres to filter small bboxes. (default: %(default)s)')
    parser.add_argument('-d', "--device", dest="device",
                        choices=["cpu", "gpu"], default="cpu",
                        help="Device to inference on. (default: %(default)s)")

    return parser
