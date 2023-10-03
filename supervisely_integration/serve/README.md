<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/MBPTrack3D/assets/119248312/7cf2e9df-798b-4a4f-a092-18c04101bfea"/>  

# MBPTrack 3D Point Cloud Tracking (ICCV2023)

State-of-the-art 3D single object tracking model (ICCV2023) integrated into Supervisely 3D Point Cloud Episodes labeling tool

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

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mbptrack3d/supervisely_integration/serve" src="https://github.com/supervisely-ecosystem/MBPTrack3D/assets/115161827/049135e8-2eac-43a3-a511-8da674fe551f" width="500px" style='padding-bottom: 20px'/> 

2. Or you can run the app from **Neural Networks** page -> **Point Clouds** -> **Detection & Tracking**.

<img src="https://github.com/supervisely-ecosystem/MBPTrack3D/assets/119248312/c220baeb-677a-4385-8dd9-f925bae02b41"/>  

3. Select one of the suggested checkpoints.

4. Run the app on an agent with `GPU`. For **Community Edition** - users have to run the app on their own GPU computer connected to the platform. Watch this [video tutorial](https://youtu.be/aO7Zc4kTrVg).

<img src="https://github.com/supervisely-ecosystem/MBPTrack3D/assets/119248312/2b23cfe9-7cdb-44d0-952d-a6cb47e602f7"/>

5. Use in `3D Point Cloud Episodes labeling tool`.

<img src="https://github.com/supervisely-ecosystem/MBPTrack3D/assets/119248312/ec9e47c9-8ff8-466c-83d9-10295e1c939d"/>

# How to use

0. The first step is to familiarise yourself with the use of the [3D Episodes Labeling Toolbox](https://app.supervisely.com/ecosystem/annotation_tools/pointcloud-episodes-labeling-tool). More information and instructions can be found [HERE](https://supervise.ly/labeling-toolbox/3d-lidar-sensor-fusion?_ga=2.243685765.1054711181.1696213910-1002110389.1685351840).

1. Create classes with Cuboid or Point Cloud shapes and then draw figures on the selected frame. There can be multiple figure per object (class) in a frame.

2. Choose the start frame, in track settings select running MBPTrack app, direction, and number of frames.

3. Click `Track` button. When a figure on the starting frame is selected, tracking begins for that figure. If no figures are selected, tracking starts for all of the figures on the frame. You can correct the position of the figures and re-track them.

 
# Demo

Here is an example of tracking cuboids on multiple frames via MBPTrack:

https://user-images.githubusercontent.com/119248312/8a250cd6-891a-46ad-9650-f8e237c9db9f.mp4

https://user-images.githubusercontent.com/119248312/c7b605d7-d56a-4222-9f2e-1d326d9f001c.mp4




