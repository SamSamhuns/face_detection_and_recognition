import pytest
import numpy as np
from PIL import Image
import tensorflow as tf

np.random.seed(42)


@pytest.fixture(scope="function")
def mock_dataset(tmp_path):
    # create data dir in tmp dir
    data_root = tmp_path / "data"
    data_root.mkdir()
    for sd_id in range(10):
        sub_dir = data_root / f"class_{sd_id}"
        sub_dir.mkdir()
    return str(data_root)


@pytest.fixture(scope="function")
def mock_numpy_image_path(tmp_path):
    # create images dir in tmp dir
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_np_path = img_dir / "numpy_random_img.jpg"
    # create a random numpy image
    img_np = (np.random.randn(160, 160, 3) * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    img.save(str(img_np_path))

    return str(img_np_path)


@pytest.fixture(scope="session")
def mock_face_feature_ext_model():
    mpath = "models/facenet/facenet_keras_p38"
    model = tf.keras.models.load_model(mpath, compile=False)
    return model
