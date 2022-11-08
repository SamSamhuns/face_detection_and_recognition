import os
import json
import glob
import pickle
import mimetypes
from pathlib import Path
from typing import Union, Dict, Any
from collections import OrderedDict


def get_file_type(file_src: Union[int, str]) -> str:
    """
    Returns if a file is image/video/camera/None based on extension or int type
    """
    if file_src.isnumeric():
        return 'camera'
    mimetypes.init()
    mimestart = mimetypes.guess_type(file_src)[0]

    file_type = None
    if mimestart is not None:
        mimestart = mimestart.split('/')[0]
        if mimestart in ['video', 'image']:
            file_type = mimestart
    return file_type


def read_pickle(pickle_path: str) -> Any:
    with open(pickle_path, 'rb') as stream:
        pkl_data = pickle.load(stream)
    return pkl_data


def write_pickle(pickle_path: str, object: Any) -> None:
    with open(pickle_path, 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_json(fname: str) -> dict:
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: Dict, fname: str) -> None:
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def gen_class2label_from_dir(data_dir: str, json_path: str) -> None:
    """
    Note: all classes must be exactly one level under data_dir
          and labels will be assigned according to alphabetical order
    """
    class_list = sorted(glob.glob(os.path.join(data_dir, "*")))
    class_list = [directory for directory in class_list if os.path.isdir(directory)]
    class_label_dict = {directory.split('/')[-1]: i for i, directory in enumerate(class_list)}
    write_json(class_label_dict, json_path)
