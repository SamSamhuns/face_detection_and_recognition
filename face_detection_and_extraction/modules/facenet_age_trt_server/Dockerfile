FROM nvcr.io/nvidia/tritonserver:21.08-py3

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV CUDA_VISIBLE_DEVICES=0

WORKDIR /app

# copy model files
COPY models /app/models

# run triton server
CMD ["tritonserver", "--model-store", "/app/models", "--allow-grpc=true", "--allow-http=false", "--grpc-port=8081", "--allow-metrics=false", "--allow-gpu-metrics=false"]
