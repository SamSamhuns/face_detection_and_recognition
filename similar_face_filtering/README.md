## Extract faces from a unfiltered data that are similar to reference faces data

The facenet keras model can be downloaded from this [Google Drive Link](https://drive.google.com/file/d/1QpVqIr6dQpknZSZFEpGJn4iZnjlgwyiN/view?usp=sharing)

Sample data-structure for UNFILTERED_DATA_PATH and REFERENCE_DATA_PATH are given in `faces_reference` and `faces_unfiltered` where data is organized as:

    root
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

```shell
# to extract similar faces to a new FILTERED_DATA_PATH directory
$ python filter_faces_using_reference.py -ud <UNFILTERED_DATA_PATH> -rd <REFERENCE_DATA_PATH> -td <FILTERED_DATA_PATH>

# i.e. to run on the given images
$ python filter_faces_using_reference.py -ud face_feat_extraction_and_filtering/faces_unfiltered -rd face_feat_extraction_and_filtering/faces_reference -td faces_filtered
```
