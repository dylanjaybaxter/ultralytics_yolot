
docker build -f ./yolot_docker/Dockerfile -t yolot_trainingimage .

docker run -t -v bdd100k-data:/workspace/dataset -v bdd-100k-results:/workspace/results --gpus all --ipc=host yolot_trainingimage
