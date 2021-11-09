FROM nvcr.io/nvidia/tritonserver:21.08-py3

# ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
# ENV CUDA_VISIBLE_DEVICES=0

# install opencv reqs
RUN apt-get update \
 && apt-get install libsm6 libxext6 libgl1-mesa-glx ffmpeg -y

WORKDIR /app

# install pip requirements
RUN python3 -m pip install --upgrade pip
RUN pip install opencv-python==4.5.3.56
RUN pip install torch==1.9.1

# copy model files
COPY models /app/models

# run triton server
CMD ["tritonserver", "--model-store", "/app/models", "--allow-grpc=true", "--allow-http=false", "--grpc-port=8081", "--allow-metrics=false", "--allow-gpu-metrics=false"]
