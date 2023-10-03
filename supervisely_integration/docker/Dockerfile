FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install python3-pip -y
RUN apt-get install -y git
RUN python3 -m pip install --upgrade pip

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

RUN python3 -m pip install supervisely==6.72.145

RUN apt-get install ffmpeg libgeos-dev libsm6 libxext6 libexiv2-dev libxrender-dev libboost-all-dev -y
RUN pip install opencv-python

RUN pip3 install addict
RUN pip3 install pyquaternion
RUN pip3 install nuscenes-devkit
RUN pip3 install starlette==0.27.0
RUN pip3 install open3d==0.17.0
RUN apt-get -y install wget

RUN mkdir -p checkpoints
RUN wget https://github.com/supervisely-ecosystem/serve-segment-anything-hq/releases/download/v1.0.3/mbptrack_kitti_car.ckpt -O /checkpoints/mbptrack_kitti_car.ckpt
RUN wget https://github.com/supervisely-ecosystem/serve-segment-anything-hq/releases/download/v1.0.3/mbptrack_kitti_cyc.ckpt -O /checkpoints/mbptrack_kitti_cyc.ckpt
RUN wget https://github.com/supervisely-ecosystem/serve-segment-anything-hq/releases/download/v1.0.3/mbptrack_kitti_ped.ckpt -O /checkpoints/mbptrack_kitti_ped.ckpt
RUN wget https://github.com/supervisely-ecosystem/serve-segment-anything-hq/releases/download/v1.0.3/mbptrack_kitti_van.ckpt -O /checkpoints/mbptrack_kitti_van.ckpt
RUN wget https://github.com/supervisely-ecosystem/serve-segment-anything-hq/releases/download/v1.0.3/mbptrack_nuscenes_kitti_car.ckpt -O /checkpoints/mbptrack_nuscenes_kitti_car.ckpt
RUN wget https://github.com/supervisely-ecosystem/serve-segment-anything-hq/releases/download/v1.0.3/mbptrack_nuscenes_kitti_bus.ckpt -O /checkpoints/mbptrack_nuscenes_kitti_bus.ckpt
RUN wget https://github.com/supervisely-ecosystem/serve-segment-anything-hq/releases/download/v1.0.3/mbptrack_nuscenes_kitti_ped.ckpt -O /checkpoints/mbptrack_nuscenes_kitti_ped.ckpt
RUN wget https://github.com/supervisely-ecosystem/serve-segment-anything-hq/releases/download/v1.0.3/mbptrack_nuscenes_kitti_trailer.ckpt -O /checkpoints/mbptrack_nuscenes_kitti_trailer.ckpt
RUN wget https://github.com/supervisely-ecosystem/serve-segment-anything-hq/releases/download/v1.0.3/mbptrack_nuscenes_kitti_truck.ckpt -O /checkpoints/mbptrack_nuscenes_kitti_truck.ckpt