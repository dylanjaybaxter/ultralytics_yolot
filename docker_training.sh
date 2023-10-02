#!/bin/bash

# Build the docker
docker build -f ./yolot_docker/Dockerfile \
     --build-arg UID_VAR=$(id -u) \
     --build-arg GID_VAR=$(id -g) \
     -t yolot_trainingimage .

docker run -v bdd100k-data:/workspace/dataset \
     --gpus all \
     --ipc=host yolot_trainingimage
