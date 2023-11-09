
docker build -f ./yolot_docker/Dockerfile -t yolot_trainingimage .

docker run -it -p 6006:6006 -e MASTER_ADDR=localhost -e MASTER_PORT=12355 -v bdd100k-data:/workspace/dataset -v bdd-100k-results:/workspace/results --gpus all --ipc=host yolot_trainingimage
