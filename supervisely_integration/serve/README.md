<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/MBPTrack3D/assets/119248312/e70eb051-8a01-4394-9703-ddaf60d05627"/>  

# MBPTrack 3D Point Cloud Tracking (ICCV2023)

state-of-the-art 3D single object tracking model (ICCV2023) integrated into Supervisely 3D Point Cloud Labeling tool

<p align="center">
  <a href="#Original-work">Original work</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a> •
    <a href="#Demo">Demo</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mbptrack3d)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/mbptrack3d)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/mbptrack3d/supervisely_integration/serve.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/mbptrack3d/supervisely_integration/serve.png)](https://supervise.ly)

</div>

# Original work

Original work is available here: [**paper (ICCV2023)**](https://arxiv.org/abs/2303.05071) and [**code**](https://github.com/slothfulxtx/MBPTrack3D).

![Architecture](https://user-images.githubusercontent.com/91027877/271337328-895d7dfd-7e14-4a35-9135-6f4a354a8a5a.jpg)

## Highlights

### New approach to 3D single obejct tracking

Unlike previous existing approaches, which were based on the Siamese paradigm, MBPTrack uses both temporal and spatial contextual information in the 3D single object tracking task with the help a memory mechanism.

### State-of-the-art performance on many public benchmarks

![kitti_table](https://user-images.githubusercontent.com/91027877/271342229-b029aabb-3a66-4351-be5a-aba06ae902f7.jpg)

![nuscenes_waymo_table](https://user-images.githubusercontent.com/91027877/271340265-b69e55df-c1ac-4e72-8528-b8213795b408.jpg)

# How to run

0. This 3D single object tracking app is started by default in most cases by an instance administrator. If it isn't available in the video labeling tool, you can contact your Supervisely instance admin or run this app by yourself following the steps below.

1. Go to Ecosystem page and find the app [MBPTrack 3D Point Cloud Tracking (ICCV2023)](https://ecosystem.supervisely.com/apps/mbptrack3d/supervisely_integration/serve).  

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mbptrack3d/supervisely_integration/serve" src="XXX" width="500px" style='padding-bottom: 20px'/> 

2. Or you can run the app from **Neural Networks** page -> **Point Clouds** -> **Detection & Tracking**.

<img src="XXX"/>  

3. Run the app on an agent with `GPU`. For **Community Edition** - users have to run the app on their own GPU computer connected to the platform. Watch this [video tutorial](https://youtu.be/aO7Zc4kTrVg).

<img src="XXX"/>

5. Use in `3D Point Cloud labeling tool`.

<img src="XXX"/>

# How to use

0. The first step is to familiarise yourself with the use of the [3D Labeling Toolbox](https://app.supervisely.com/ecosystem/annotation_tools/pointcloud-labeling-tool). More information and instructions can be found [HERE](https://supervise.ly/labeling-toolbox/3d-lidar-sensor-fusion?_ga=2.243685765.1054711181.1696213910-1002110389.1685351840).
 
# Demo

Here is an example of tracking cuboids on multiple frames via MBPTrack:


