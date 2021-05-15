import os
import tensorflow as tf

warm_start_point = "facenet_keras.h5"  # name of h5 model
export_path = '1/model.savedmodel'
# set compile to False for non-training inference only model
model = tf.keras.models.load_model(os.path.join('./models/facenet', warm_start_point),
                                   compile=False)

model_in_width, model_in_height, model_in_channels = 160, 160, 3


class CustomModule(tf.Module):
    def __init__(self, model, **kwargs):
        super(CustomModule, self).__init__(**kwargs)
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,
                                                       model_in_width,
                                                       model_in_height,
                                                       model_in_channels),
                                                dtype=tf.uint8)])
    def update_signature(self, inp_images):  # inp_images is the input name
        x = tf.image.per_image_standardization(inp_images)
        return {"predictions": self.model(x)}


os.makedirs(export_path, exist_ok=True)
module = CustomModule(model)
tf.saved_model.save(module,
                    os.path.join(export_path),
                    signatures={"serving_default": module.update_signature})
