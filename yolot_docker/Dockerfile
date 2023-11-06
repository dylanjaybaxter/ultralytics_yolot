
FROM pytorch/pytorch

# Import Requirements
RUN pip install --upgrade pip
RUN pip install ultralytics
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
# RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python
RUN pip install tensorboard


# Setup Work Directory
WORKDIR /workspace

# Copy Files to Working Directory
RUN rm -rf /workspace
COPY . /workspace
RUN pip install -r requirements.txt
RUN pip install --upgrade torchmetrics

# When started run the training
CMD python3 -m torch.distributed.run --nnodes=1 --nproc_per_node=2 train_yolot.py --conf yolot_config_unix.yaml