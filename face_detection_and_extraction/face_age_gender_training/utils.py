import json
import glob
import pickle
from pathlib import Path
from itertools import repeat
from datetime import datetime
from collections import OrderedDict

import torch
import numpy as np
import pandas as pd
from scipy.io import loadmat


def read_pickle(pickle_path: str):
    with open(pickle_path, 'rb') as stream:
        foo = pickle.load(stream)
    return foo


def update_lr_scheduler(config: dict, train_dataloader_size: int):
    """
    Update the key values in the dict so that the lr_scheduler is compatible
    with the optimizer and trainer

    config: ConfigParser object
    """
    if config['lr_scheduler']['type'] == "OneCycleLR":
        config['lr_scheduler']['args'].update(
            {"max_lr": config['optimizer']['args']['lr']})

        config['lr_scheduler']['args'].update(
            {"epochs": config['trainer']['epochs']})

        config['lr_scheduler']['args'].update(
            {"steps_per_epoch": train_dataloader_size})

    return config


def get_original_xy(xy: tuple, image_size_original: tuple, image_size_new: tuple) -> tuple:
    """Every point is from the PIL Image point of view.

    Note that image_size_original and image_size_new are PIL Image size values.
    """
    x, y = xy[0], xy[1]

    width_original = image_size_original[0]
    height_original = image_size_original[1]

    width_new = image_size_new[0]
    height_new = image_size_new[1]

    if width_original > height_original:

        x = width_original / width_new * x
        y = width_original / width_new * y

        bias = (width_original - height_original) / 2
        y = y - bias

    else:

        x = height_original / height_new * x
        y = height_original / height_new * y

        bias = (height_original - width_original) / 2
        x = x - bias

    return x, y


def get_original_bbox(bbox: np.ndarray, image_size_original: tuple, image_size_new: tuple) -> np.ndarray:
    """Get the original coordinates of the bounding box.

    Note that image_size_original and image_size_new are PIL Image size values.
    """
    bbox_new = []
    for xy in [bbox[:2], bbox[2:]]:
        xy = xy[0], xy[1]
        xy = get_original_xy(xy, image_size_original, image_size_new)
        bbox_new.append(xy)

    bbox_new = [bar for foo in bbox_new for bar in foo]
    bbox_new = np.array(bbox_new)

    return bbox_new


def get_original_lm(lm: np.ndarray, image_size_original: tuple, image_size_new: tuple) -> np.ndarray:
    """Get the original coordinates of the five landmarks.

    Note that image_size_original and image_size_new are PIL Image size values.
    """
    lm_new = []
    for lm_ in lm:
        xy = lm_[0], lm_[1]
        xy = get_original_xy(xy, image_size_original, image_size_new)
        lm_new.append(xy)

    lm_new = np.array(lm_new)

    return lm_new


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(
            index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / \
            self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def calc_age(taken, dob):
    """Copied from
    https://github.com/yu4u/age-gender-estimation/blob/master/src/utils.py
    """
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def get_metadata_from_imdb_wiki(mat_path, db):
    """Copied from
    https://github.com/yu4u/age-gender-estimation/blob/master/src/utils.py
    """
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    assert len(full_path) == len(dob) == len(gender) == len(
        photo_taken) == len(face_score) == len(second_face_score) == len(age)
    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def load_data(mat_path):
    """Copied from
    https://github.com/yu4u/age-gender-estimation/blob/master/src/utils.py
    """
    d = loadmat(mat_path)

    return d["image"], d["gender"][0], d["age"][0], d["db"][0], d["img_size"][0, 0], d["min_score"][0, 0]


def recursively_get_file_paths(root_dir, ext="npy"):
    """
    Get file paths recursively in the 'root_dir' with the extension 'ext'
    """
    fpaths = []
    for fpath in glob.glob(f"{root_dir}/**/*.{ext}", recursive=True):
        fpaths.append(fpath)
    return fpaths
