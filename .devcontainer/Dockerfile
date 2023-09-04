FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install python3-pip -y
RUN apt-get install -y git

RUN pip3 install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip3 install pytorch-lightning==1.5.10
RUN pip3 install -U 'git+https://github.com/facebookresearch/fvcore'
RUN pip3 install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
RUN git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
WORKDIR Pointnet2_PyTorch
RUN pip3 install pointnet2_ops_lib/.
WORKDIR /
RUN pip3 install shapely==1.8.1

RUN python3 -m pip install supervisely==6.72.87

RUN apt-get install ffmpeg libgeos-dev libsm6 libxext6 libexiv2-dev libxrender-dev libboost-all-dev -y
RUN pip install opencv-python

RUN pip3 install addict
RUN pip3 install pyquaternion
RUN apt-get -y install curl

RUN apt -y install wireguard iproute2
RUN apt-get -y install wget
RUN pip3 install nuscenes-devkit
RUN apt-get -y install unzip
RUN pip3 install starlette==0.27.0
RUN pip3 install open3d