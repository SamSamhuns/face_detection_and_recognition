import os
# os environments must be set at the beginning of the file top use GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["OMP_NUM_THREADS"] = "15"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# specify which GPU(s) to be used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import glob
import shutil
import argparse
from typing import List, Tuple

import tqdm
import numpy as np
import tensorflow as tf

# fix seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def _fix_path_for_globbing(dir: str) -> str:
    """ Add * at the end of paths for proper globbing
    """
    if dir[-1] == '/':         # data/
        dir += '*'
    elif dir[-1] != '*':       # data
        dir += '/*'
    else:                      # data/*
        dir = dir

    return dir


def get_class_name_list(base_dir: str) -> List[str]:
    """ base_dir struct
        data
            |_class1
                    |_ c1img1
                    |_ c1img2
                    |_ ...
            |_class2
                    |_ c2img1
                    |_ c2img2
                    |_ ...
    """
    map_list = []
    subdir_list = sorted(glob.glob(_fix_path_for_globbing(base_dir)))
    for class_dir in subdir_list:
        map_list.append(class_dir.split('/')[-1])
    return map_list


def read_and_preprocess_img(img_path: str,
                            in_size: Tuple[int, int] = (160, 160),
                            dct_method: str = "INTEGER_FAST") -> tf.Tensor:
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method=dct_method)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=in_size)
    img = tf.image.per_image_standardization(img)  # standardize per channel
    return img


def get_ref_mean_vec_and_thres_from_imgs(model: tf.keras.Model,
                                         ref_class_path: str,
                                         max_ref_img_count: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Get ref mean_vec and thres from a give ref class path
    """
    X_imgs = glob.glob(ref_class_path + "/*.jpg")
    ref_dataset = tf.data.Dataset.from_tensor_slices((X_imgs))
    ref_dataset = ref_dataset.map(
        lambda x: (read_and_preprocess_img(x))).batch(1)

    ref_num = min(max_ref_img_count, len(ref_dataset))
    ref_feat = []
    for i, img_batch in enumerate(ref_dataset):
        if i >= max_ref_img_count:
            break
        ref_feat.append(model.predict(img_batch, verbose=0))
    ref_feat = np.asarray(ref_feat)
    ref_mean_vec = np.mean(ref_feat, axis=0)

    # calc max dist from mean vector
    max_dist_from_mean = 0
    for i in range(ref_num):
        max_dist_from_mean = max(max_dist_from_mean,
                                 np.linalg.norm(ref_mean_vec - ref_feat[i]))

    print(f"number of samples considered for reference={ref_num}",
          f"ref mean shape={ref_mean_vec.shape}",
          f"ref feat shape={ref_feat.shape}")
    print("max dist from mean in the reference batch: ", max_dist_from_mean)

    thres = max_dist_from_mean  # thres for one class cls with l2 norm
    return ref_mean_vec, thres


def get_parsed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ud', '--unfiltered_data_path', dest="unfiltered_data_path",
                        type=str, required=True,
                        help='Unfiltered raw face dataset path with class imgs in subdirs')
    parser.add_argument('--rd', '--reference_data_path', dest="reference_data_path",
                        type=str, required=True,
                        help='Reference face dataset path with class imgs in each subdirs that are manually prefiltered')
    parser.add_argument('--td', '--target_data_path', dest="target_data_path",
                        type=str, default="data/faces_filtered",
                        help='Dataset path where subdirs clean and unclean contain respective filtered classes')
    parser.add_argument('-m', '--savedmodel_path',
                        type=str, default="models/facenet/facenet_keras_p38",
                        help='Path to savedmodel feat feature extraction model. (default: %(default)s)')
    parser.add_argument('-b', '--batch_size',
                        type=int, default=32,
                        help='Dataloader batch size. (default: %(default)s)')
    parser.add_argument('-r', '--ref_img_per_class',
                        type=int, default=32,
                        help='Number of reference images to consider per class: %(default)s)')
    args = parser.parse_args()
    return args


def main():
    args = get_parsed_args()
    print(args)

    model = tf.keras.models.load_model(args.savedmodel_path, compile=False)

    # printing input and output shape
    print(f"Printing signature of model from {args.savedmodel_path}")
    print("\tInput:", model.inputs)
    print("\tOutput:", model.outputs)

    UNFILTERED_ROOT = _fix_path_for_globbing(args.unfiltered_data_path)
    REFERENCE_ROOT = _fix_path_for_globbing(args.reference_data_path)
    TARGET_ROOT = args.target_data_path

    ref_class_paths = glob.glob(REFERENCE_ROOT)
    unfiltered_class_paths = glob.glob(UNFILTERED_ROOT)

    if len(unfiltered_class_paths) != len(ref_class_paths):
        raise Exception(f"Class number Error. \
            Unfiltered root {UNFILTERED_ROOT} \
            and reference root {REFERENCE_ROOT} \
            must have the same number of classes")
    for i in range(len(ref_class_paths)):
        if ref_class_paths[i].split('/')[-1] != unfiltered_class_paths[i].split('/')[-1]:
            raise Exception("class {} and {} did not match")

    # create target dirs
    clean_dir = os.path.join(TARGET_ROOT, 'clean')
    unclean_dir = os.path.join(TARGET_ROOT, 'unclean')
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(unclean_dir, exist_ok=True)

    # iterate through each reference dir class
    for i, ref_class_path in enumerate(tqdm.tqdm(ref_class_paths)):
        print(f"Calculating ref mean vector for {ref_class_path}")
        ref_mean_vec, thres = get_ref_mean_vec_and_thres_from_imgs(
            model, ref_class_path, max_ref_img_count=args.ref_img_per_class)

        unfiltered_class_path = unfiltered_class_paths[i]

        X_imgs = glob.glob(unfiltered_class_path + "/*.jpg")
        unfiltered_dataset = tf.data.Dataset.from_tensor_slices((X_imgs))
        unfiltered_dataset = unfiltered_dataset.map(
            lambda x: (read_and_preprocess_img(x))).batch(args.batch_size)

        img_cnt = total = similar_cnt = 0

        filtered_class_clean_dir = os.path.join(
            clean_dir, unfiltered_class_path.split('/')[-1])
        filtered_class_unclean_dir = os.path.join(
            unclean_dir, unfiltered_class_path.split('/')[-1])
        # make class dirs
        os.makedirs(filtered_class_clean_dir, exist_ok=True)
        os.makedirs(filtered_class_unclean_dir, exist_ok=True)

        for img_batch in unfiltered_dataset:
            output_batch = model.predict(img_batch, verbose=0)
            total += output_batch.shape[0]
            for i, out in enumerate(output_batch):
                class_plus_img_name = X_imgs[img_cnt].split('/')[-1]
                # images are similar using the euclidean distance metric
                if np.linalg.norm(out - ref_mean_vec) <= thres:
                    similar_cnt += 1
                    target_path = os.path.join(
                        filtered_class_clean_dir, class_plus_img_name)
                else:
                    target_path = os.path.join(
                        filtered_class_unclean_dir, class_plus_img_name)
                shutil.copy(X_imgs[img_cnt], target_path)
                img_cnt += 1
        print(
            f"Similar images percentage={similar_cnt/total:2.2f}%, positive={similar_cnt}, total={total}")


if __name__ == "__main__":
    main()
