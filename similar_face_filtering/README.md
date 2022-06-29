# Similar Face Filtering

[![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)](https://www.python.org/downloads/release/python-380/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)](https://www.python.org/downloads/release/python-390/)

Extract faces from an unfiltered and noisy face dataset that are similiar to a reference face dataset, avoiding the dissimilar and noisy images.

The facenet keras model used for filtering is available [here](https://drive.google.com/drive/folders/1juZ6vj9eUUdam-RtH36nTdzBnoinsWcy?usp=sharing). The `facenet_keras_p38` should be unzipped and placed in `similar_face_filtering/models/facenet`.

The faces must be kept inside directories with the classnames. A sample data structure for `UNFILTERED_DATA_PATH` and `REFERENCE_DATA_PATH` are provided in `data/faces_reference` and `data/faces_unfiltered` respectively, that have the following organization:

    data_dir
      |_ class1
        |_ img11
        |_ img12
        |_ img13
        |_ ...
      |_ class2
        |_ img21
        |_ img22
        |_ img23
        |_ ...
      |_ ...

## Extract similar faces into a filtered directory

```shell
# to extract similar faces to a new FILTERED_DATA_PATH directory
$ python filter_faces_using_reference.py \
        --ud UNFILTERED_DATA_PATH \
        --rd REFERENCE_DATA_PATH \
        --td FILTERED_DATA_PATH \
        -b BATCH_SIZE \
        -r REFERENCE_IMGS_PER_CLASS

# i.e. to run on the images inside data dir
$ python filter_faces_using_reference.py \
      --ud data/faces_unfiltered \
      --rd data/faces_reference \
      --td data/faces_filtered \
      -b 32 \
      -r 32
```

## Run Tests

Install requirements

```shell
pip install tests/requirements-dev.txt
```

Run tests from inside the `similar_face_filtering directory`

```shell
PYTHONPATH=$PYTHONPATH:./ pytest tests
```
