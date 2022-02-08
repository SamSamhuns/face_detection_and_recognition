import numpy as np
from tritonclient import grpc as grpcclient
from tritonclient.utils import InferenceServerException


class FlagConfig:
    """Stores configurations for inference"""

    def __init__(self):
        pass


def get_client_and_model_metadata_config(FLAGS: FlagConfig):
    # Create gRPC client for communicating with the server
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url, verbose=FLAGS.verbose)
    except Exception as e:
        print("client creation failed: " + str(e))
        return -1
    # Get model metadata i.e. input, output shapes
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        return -1
    # Get model config
    try:
        model_config = triton_client.get_model_config(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        return -1

    return triton_client, model_metadata, model_config


def requestGenerator(input_data_list, input_name_list, output_name_list, input_dtype_list, FLAGS):

    """set inputs and outputs"""
    inputs = []
    for i in range(len(input_name_list)):
        inputs.append(grpcclient.InferInput(
            input_name_list[i], input_data_list[i].shape, input_dtype_list[i]))
        inputs[i].set_data_from_numpy(input_data_list[i])

    outputs = []
    for i in range(len(output_name_list)):
        outputs.append(grpcclient.InferRequestedOutput(
            output_name_list[i], class_count=FLAGS.classes))

    yield inputs, outputs, FLAGS.model_name, FLAGS.model_version


def parse_model_grpc(model_metadata, model_config):

    input_format_list = []
    input_datatype_list = []
    input_metadata_name_list = []
    for i in range(len(model_metadata.inputs)):
        input_format_list.append(model_config.input[i].format)
        input_datatype_list.append(model_metadata.inputs[i].datatype)
        input_metadata_name_list.append(model_metadata.inputs[i].name)
    output_metadata_name_list = []
    for i in range(len(model_metadata.outputs)):
        output_metadata_name_list.append(model_metadata.outputs[i].name)
    # the first input must always be the image array / batch dimension
    s1 = model_metadata.inputs[0].shape[1]
    s2 = model_metadata.inputs[0].shape[2]
    s3 = model_metadata.inputs[0].shape[3]
    return (model_config.max_batch_size, input_metadata_name_list,
            output_metadata_name_list, s1, s2, s3, input_format_list,
            input_datatype_list)


def get_inference_responses(image_data_list, FLAGS, trt_inf_data):
    triton_client, input_name, output_name, input_dtype, max_batch_size = trt_inf_data
    responses = []
    image_idx = 0
    sent_count = 0
    last_request = False

    image_data = image_data_list[0]
    while not last_request:
        repeated_image_data = []
        for idx in range(FLAGS.batch_size):
            repeated_image_data.append(image_data[image_idx])
            image_idx = (image_idx + 1) % len(image_data)
            if image_idx == 0:
                last_request = True
        if max_batch_size > 0:
            batched_image_data = np.stack(repeated_image_data, axis=0)
        else:
            batched_image_data = repeated_image_data[0]
        if max_batch_size == 0:
            batched_image_data = np.expand_dims(batched_image_data, 0)

        input_image_data = [batched_image_data]
        # if more inputs are present
        # then add other inputs to input_image_data
        if len(image_data_list) > 1:
            for in_data in image_data_list[1:]:
                input_image_data.append(in_data)
        # Send request
        try:
            for inputs, outputs, model_name, model_version in requestGenerator(
                    input_image_data, input_name, output_name, input_dtype, FLAGS):
                sent_count += 1
                responses.append(
                    triton_client.infer(FLAGS.model_name,
                                        inputs,
                                        request_id=str(sent_count),
                                        model_version=FLAGS.model_version,
                                        outputs=outputs))
        except InferenceServerException as e:
            print("inference failed: " + str(e))
            return -1

    return responses
