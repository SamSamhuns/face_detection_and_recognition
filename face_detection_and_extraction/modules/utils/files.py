import json
import pickle
import mimetypes
from pathlib import Path
from typing import Union, Dict
from collections import OrderedDict


def get_file_type(file_src: Union[int, str]):
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


def read_pickle(pickle_path: str):
    with open(pickle_path, 'rb') as stream:
        pkl_data = pickle.load(stream)
    return pkl_data


def read_json(fname: str):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: Dict, fname: str):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
