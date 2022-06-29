from PIL import Image
import numpy as np

from filter_faces_using_reference import _fix_path_for_globbing, get_class_name_list
from filter_faces_using_reference import read_and_preprocess_img, get_ref_mean_vec_and_thres_from_imgs


def test_fix_path_for_globbing():
    assert _fix_path_for_globbing("data/"), "data/*"
    assert _fix_path_for_globbing("data"), "data/*"
    assert _fix_path_for_globbing("data/*"), "data/*"


def test_get_class_name_list(mock_dataset):
    assert get_class_name_list(mock_dataset) == [
        f"class_{i}" for i in range(10)]


def test_read_and_preprocess_img(mock_numpy_image_path):
    img_np_path = mock_numpy_image_path
    img_np = np.array(Image.open(img_np_path))
    tf_img_np = read_and_preprocess_img(
        img_np_path, in_size=(160, 160), dct_method="INTEGER_ACCURATE").numpy()
    N = img_np.shape[0] * img_np.shape[1] * img_np.shape[2]
    img_np_std = (img_np - np.mean(img_np)) / \
        max(np.std(img_np), 1 / (N ** (1 / 2)))
    assert np.allclose(tf_img_np, img_np_std, atol=0.0001)


def test_get_ref_mean_vec_and_thres_from_imgs_facenet(mock_face_feature_ext_model):
    model = mock_face_feature_ext_model
    ref_data_dir_path = "data/faces_reference/LN-MANUEL_MIRANDA"

    ref_mean_vec, thres = get_ref_mean_vec_and_thres_from_imgs(
        model, ref_data_dir_path)
    thres_ref = 7.5812364
    ref_mean_vec_ref = np.array([[0.03088139, -0.50232756, -1.4322278, -0.4012962, 0.98117083,
                                  -0.9980548, 0.74885494, 0.83298385, -0.28337336, 0.4920226,
                                  0.04177739, -0.8087803, 0.44652912, -0.36705622, 0.7899383,
                                  2.2632635, -0.06147028, -0.09982201, -0.46557507, 0.57861567,
                                  0.4988199, 0.00939741, 0.30188054, 1.8760308, -0.5504848,
                                  -0.9258094, 1.3077208, -0.56640255, -1.2653025, 0.8767949,
                                  -0.04382955, 0.4266773, 0.15036544, 1.2829263, -0.03089689,
                                  -0.7457357, -1.050124, -0.9489217, 0.6708629, 1.378409,
                                  -0.22471948, -0.77625954, 2.1669166, 0.7725577, 0.59916943,
                                  -1.2560791, 0.37818214, 0.05331777, 0.5230588, 0.7613669,
                                  -0.86749125, 0.26470944, -0.38126844, 0.78634775, 0.22124423,
                                  -1.3393502, -0.9408233, 0.19477923, -1.516134, 0.70708495,
                                  -1.1873397, -0.41912884, 0.21431844, -0.7283546, 0.2588193,
                                  -0.65422535, -0.647267, 0.26367217, -0.43967083, -0.46330652,
                                  0.02885669, 0.57527745, 0.65981764, 0.6012105, -0.9470239,
                                  -1.3792024, -0.23033209, -0.94295704, 0.32888007, -0.5894558,
                                  -0.30252445, 0.81183904, 1.3326955, 0.49669236, 0.65110314,
                                  0.6386908, -0.2421939, -0.8499881, 0.14156196, 0.27745363,
                                  0.40803793, 0.7954835, -0.14072545, 1.2041621, -0.0168117,
                                  -0.39692235, -1.1121427, 0.07845795, -1.7255399, 0.6096816,
                                  0.6844979, -1.0797906, -0.21917294, 0.9024277, -1.0531561,
                                  -2.2125537, -0.5988225, 0.44821763, -2.1634016, 0.42769283,
                                  0.97734505, -0.40714195, 2.2259839, 0.15541187, -0.51696527,
                                  -2.046331, 1.4172497, 0.34327555, 0.37885353, -0.6521776,
                                  0.95489234, 0.5727488, -0.03939377, -0.5176346, -2.2069402,
                                  -0.13185886, -0.6810523, -0.6597599]])
    assert np.allclose(thres, thres_ref, atol=0.01)
    assert np.allclose(ref_mean_vec, ref_mean_vec_ref, atol=0.001)
