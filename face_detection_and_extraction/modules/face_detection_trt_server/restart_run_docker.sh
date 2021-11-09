#!/bin/bash

# check for 2 cmd args
if [ $# -ne 2 ]
  then
    echo "GRPC port must be specified for tritonserver."
		echo "eg. \$ bash restart_run_docker.sh -g 8080"
		exit
fi

# get the grpc port
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -g|--grpc) grpc="$2"; shift ;;
        *) echo "Unknown parameter passed: $1";
	exit 1 ;;
    esac
    shift
done

echo "Stopping docker container 'face_det' if it is running"
docker stop face_det || true
docker rm face_det || true

echo "Running docker with exposed triton-server GRPC port: $grpc"
docker run --rm \
      --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
      --name face_det \
      -p $grpc:8081 \
      yolov5_face_detection:latest
