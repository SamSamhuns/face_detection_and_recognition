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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # , 1, 2, 3, 4, 5, 6, 7

import glob
import shutil
import argparse
import numpy as np
import tensorflow as tf


def _fix_path_for_globbing(dir):
    """ Add * at the end of paths for proper globbing
    """
    if dir[-1] == '/':         # data/
        dir += '*'
    elif dir[-1] != '*':       # data
        dir += '/*'
    else:                      # data/*
        dir = dir

    return dir


def get_class_name_list(base_dir):
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
    subdir_list = glob.glob(_fix_path_for_globbing(base_dir))
    for class_dir in subdir_list:
        map_list.append(class_dir.split('/')[-1])
    return map_list


def get_ref_mean_vec_and_thres_from_imgs(model, ref_class_path, b_size=32):
    """Get ref mean_vec and thres from a give ref class path
    """
    X_imgs = glob.glob(ref_class_path + "/*.jpg")
    ref_dataset = tf.data.Dataset.from_tensor_slices((X_imgs))
    ref_dataset = ref_dataset.map(
        lambda x: (preprocess_img(x))).batch(b_size)

    data_iter = iter(ref_dataset)
    image_batch1 = next(data_iter)

    # standardize per channel
    image_batch1 = tf.image.per_image_standardization(image_batch1)
    ref_feat = model.predict(image_batch1)  # output_batch1
    ref_mean_vec = np.mean(ref_feat, axis=0)

    # calc max dist from mean vector
    max_dist_from_mean = 0
    for i in range(b_size):
        max_dist_from_mean = max(max_dist_from_mean,
                                 np.linalg.norm(ref_mean_vec - ref_feat[i]))

    print(f"ref mean shape={ref_mean_vec.shape}",
          f"ref feat shape={ref_feat.shape}")
    print("max dist from mean in the reference batch: ", max_dist_from_mean)

    thres = max_dist_from_mean  # thres for one class cls with l2 norm
    return ref_mean_vec, thres


def get_ref_mean_vec_and_thres(model, ref_img_dir, b_size=32):
    # img input dim set to 160,160 for the facenet h5 model
    ref_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        ref_img_dir,
        labels='inferred',
        class_names=get_class_name_list(ref_img_dir),
        image_size=(160, 160),
        seed=42,
        shuffle=True,
        batch_size=b_size)

    data_iter = iter(ref_dataset)
    image_batch1, labels_batch1 = next(data_iter)

    # standardize per channel
    image_batch1 = tf.image.per_image_standardization(image_batch1)
    ref_feat = model.predict(
        tf.cast(image_batch1, tf.float32))  # output_batch1
    ref_mean_vec = np.mean(ref_feat, axis=0)

    max_dist_from_mean = 0
    for i in range(b_size):
        max_dist_from_mean = max(max_dist_from_mean,
                                 np.linalg.norm(ref_mean_vec - ref_feat[i]))

    print(f"ref mean shape={ref_mean_vec.shape}",
          f"ref feat shape={ref_feat.shape}")
    print("max dist from mean in the reference batch: ", max_dist_from_mean)

    thres = max_dist_from_mean  # thres for one class cls with l2 norm
    return ref_mean_vec, thres


def preprocess_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=(160, 160))
    return img


def main():
    """ unfiltered_face_data_path struct
        filtered_face_data_path struct
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-ud', '--unfiltered_face_data_path',
                        type=str, required=True,
                        help='Unfiltered raw face dataset path with class imgs in subdirs')
    parser.add_argument('-rd', '--ref_face_data_path',
                        type=str, required=True,
                        help='Reference face dataset path with class imgs in each subdirs that are manually prefiltered')
    parser.add_argument('-td', '--target_data_path',
                        type=str, required=False,
                        help='Dataset path where subdirs clean and unclean contain respective filtered classes')
    parser.add_argument('-m', '--h5_model_path',
                        type=str,
                        default="models/facenet/facenet_keras.h5",
                        help='Path to h5 feat feature extraction model. (default: %(default)s)')
    args = parser.parse_args()
    model = tf.keras.models.load_model(args.h5_model_path, compile=False)

    # printing input and output shape
    print(f"Printing signature of model from {args.h5_model_path}")
    print("\tInput:", model.inputs)
    print("\tOutput:", model.outputs)

    UNFILTERED_ROOT = _fix_path_for_globbing(args.unfiltered_face_data_path)
    REFERENCE_ROOT = _fix_path_for_globbing(args.reference_face_data_path)
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
    clean_path = os.path.join(TARGET_ROOT, 'clean')
    unclean_path = os.path.join(TARGET_ROOT, 'unclean')
    os.makedirs(clean_path, exist_ok=True)
    os.makedirs(unclean_path, exist_ok=True)

    unfiltered_data_batch_size = 32
    # iterate through reference dir classes
    for i, ref_class_path in enumerate(ref_class_paths):
        print(f"Calculating ref mean vector for {ref_class_path}")
        ref_mean_vec, thres = get_ref_mean_vec_and_thres_from_imgs(
            model, ref_class_path)

        unfiltered_class_path = unfiltered_class_paths[i]

        X_imgs = glob.glob(unfiltered_class_path + "/*.jpg")
        unfiltered_dataset = tf.data.Dataset.from_tensor_slices((X_imgs))
        unfiltered_dataset = unfiltered_dataset.map(
            lambda x: (preprocess_img(x))).batch(unfiltered_data_batch_size)

        counter, total, positive = 0, 0, 0

        filtered_class_clean_dir = os.path.join(
            clean_path, unfiltered_class_path.split('/')[-1])
        filtered_class_unclean_dir = os.path.join(
            unclean_path, unfiltered_class_path.split('/')[-1])
        os.makedirs(filtered_class_clean_dir, exist_ok=True)
        os.makedirs(filtered_class_unclean_dir, exist_ok=True)

        for img_batch in unfiltered_dataset:
            # standardize per channel
            img_batch_whitened = tf.image.per_image_standardization(img_batch)
            output_batch = model.predict(img_batch_whitened)
            total += output_batch.shape[0]
            for i, out in enumerate(output_batch):
                print(X_imgs[counter])
                print(os.path.join(filtered_class_clean_dir, f"{counter}.jpg"))
                print(os.path.join(
                    filtered_class_unclean_dir, f"{counter}.jpg"))
                if np.linalg.norm(out - ref_mean_vec) <= thres:
                    positive += 1
                    shutil.copy(X_imgs[counter], os.path.join(filtered_class_clean_dir,
                                                              f"{counter}.jpg"))
                else:
                    shutil.copy(X_imgs[counter], os.path.join(filtered_class_unclean_dir,
                                                              f"{counter}.jpg"))
                counter += 1
        print(
            f"Positive percentage={positive/total:2.2f}%, positive={positive}, total={total}")


if __name__ == "__main__":
    main()
