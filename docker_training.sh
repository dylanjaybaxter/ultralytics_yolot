#!/bin/bash

# Build the docker
docker build -f ./yolot_docker/Dockerfile \
     --build-arg UID_VAR=$(id -u) \
     --build-arg GID_VAR=$(id -g) \
     -t yolot_trainingimage .

docker run -v bdd100k-data:/workspace/dataset \
    -it -p 6006:6006 -e MASTER_ADDR=localhost -e MASTER_PORT=12355 \
    -v bdd100k-results:/workspace/results \
     --gpus all \
     --ipc=host yolot_trainingimage
