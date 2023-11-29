import supervisely as sly
from supervisely.project.project_type import ProjectType
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)
from supervisely.video_annotation.video_tag_collection import VideoTagCollection
from dotenv import load_dotenv
from datasets.utils import BoundingBox, PointCloud
import open3d
import numpy as np
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d
from datasets.utils.pcd_utils import crop_and_center_pcd
import os
from tqdm import tqdm

load_dotenv("supervisely.env")
api = sly.Api()


def preprocess_cuboid(geometry: Cuboid3d):
    assert geometry.geometry_name() == "cuboid_3d"
    position = geometry.position
    rotation = geometry.rotation
    dimensions = geometry.dimensions

    rot = Rotation.from_rotvec([0, 0, rotation.z + (np.pi / 2)])
    rot_mat = rot.as_matrix()
    center = [position.x, position.y, position.z]
    size = [dimensions.x, dimensions.y, dimensions.z]
    orientation = Quaternion(matrix=rot_mat)
    return BoundingBox(center, size, orientation)


def add_padding_to_cuboid(geometry: Cuboid3d, padding: float):
    assert geometry.geometry_name() == "cuboid_3d"
    position = geometry.position
    rotation = geometry.rotation
    dimensions = geometry.dimensions

    rot = Rotation.from_rotvec([0, 0, rotation.z + (np.pi / 2)])
    rot_mat = rot.as_matrix()
    center = [position.x, position.y, position.z]
    size = [
        padding * dimensions.x,
        padding * dimensions.y,
        padding * dimensions.z,
    ]
    orientation = Quaternion(matrix=rot_mat)
    return BoundingBox(center, size, orientation)


def postprocess_cuboid(box: BoundingBox):
    position = box.center
    position = Vector3d(position[0], position[1], position[2])
    dimensions = Vector3d(box.wlh[0], box.wlh[1], box.wlh[2])
    rot = Rotation.from_matrix(box.rotation_matrix)
    rot_vec = rot.as_rotvec()
    rotation = Vector3d(0, 0, rot_vec[2] - (np.pi / 2))
    return Cuboid3d(position, rotation, dimensions)


def preprocess_pcd(pcd_path):
    pcd = open3d.io.read_point_cloud(pcd_path, format="pcd")
    points = np.asarray(pcd.points, dtype=np.float32)
    pcd = PointCloud(points.T)
    return pcd


kitti_dataset_id = 80262
kitti_pcd_infos = api.pointcloud.get_list(kitti_dataset_id)
kitti_pcd_ids = [info.id for info in kitti_pcd_infos]
lyft_dataset_id = 80815
lyft_pcd_infos = api.pointcloud.get_list(lyft_dataset_id)
lyft_pcd_ids = [info.id for info in lyft_pcd_infos]
pcd_ids = kitti_pcd_ids + lyft_pcd_ids
pcd_counter = 0

result_project = api.project.create(
    657, "Cropped point clouds", change_name_if_conflict=True, type=ProjectType.POINT_CLOUDS
)
result_dataset = api.dataset.create(result_project.id, "dataset_0", change_name_if_conflict=True)
obj_classes = sly.ObjClassCollection([sly.ObjClass("car", Cuboid3d)])
project_meta = sly.ProjectMeta(obj_classes=obj_classes)
api.project.update_meta(result_project.id, project_meta.to_json())

for pcd_id in tqdm(pcd_ids):
    ann_info = api.pointcloud.annotation.download(pcd_id)
    if len(ann_info["frames"]) < 1:
        continue
    pcd_path = f"./pointclouds/{pcd_id}.pcd"
    api.pointcloud_episode.download_path(pcd_id, pcd_path)
    figure_ids = [figure["id"] for figure in ann_info["frames"][0]["figures"]]
    geometries = []
    for figure_id in figure_ids:
        figure = api.pointcloud.figure.get_info_by_id(figure_id)
        geometry = sly.deserialize_geometry(figure.geometry_type, figure.geometry)
        geometries.append(geometry)

    input_pcd = preprocess_pcd(pcd_path)
    for geometry in geometries:
        bbox = preprocess_cuboid(geometry)
        padding = np.random.uniform(1.5, 2.5)
        padded_bbox = add_padding_to_cuboid(geometry, padding=padding)
        cropped_pcd = crop_and_center_pcd(
            input_pcd,
            padded_bbox,
        )
        _, bbox = crop_and_center_pcd(
            input_pcd,
            bbox,
            return_box=True,
        )
        new_pcd = open3d.geometry.PointCloud()
        new_pcd.points = open3d.utility.Vector3dVector(np.asarray(cropped_pcd.points).T)
        save_pcd_dir = "./cropped_pcds/"
        if not os.path.exists(save_pcd_dir):
            os.mkdir(save_pcd_dir)
        pcd_name = f"{pcd_counter}.pcd"
        save_pcd_path = os.path.join(save_pcd_dir, pcd_name)
        open3d.io.write_point_cloud(save_pcd_path, new_pcd)
        # print(f"Center: {bbox.center}")
        # print(f"Width: {bbox.wlh[0]}, Length: {bbox.wlh[1]}, Height: {bbox.wlh[2]}")
        # rot = Rotation.from_matrix(bbox.rotation_matrix)
        # rot_vec = rot.as_rotvec()
        # print(f"Rotation: {[0, 0, rot_vec[2]]}")
        # print("----------------------")
        # print()
        uploaded_pcd_info = api.pointcloud.upload_path(
            result_dataset.id, name=pcd_name, path=save_pcd_path
        )
        pcd_counter += 1
        result_cuboid = postprocess_cuboid(bbox)
        pcd_object = sly.PointcloudObject(project_meta.get_obj_class("car"))
        pcd_figure = sly.PointcloudFigure(pcd_object, result_cuboid)
        pcd_objects = PointcloudObjectCollection([pcd_object])
        pcd_figures = [pcd_figure]
        result_ann = sly.PointcloudAnnotation(pcd_objects, pcd_figures, VideoTagCollection([]))
        api.pointcloud.annotation.append(uploaded_pcd_info.id, result_ann)
