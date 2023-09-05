import numpy as np
from typing import Generator, Optional, List, Tuple, OrderedDict, Dict
from collections import OrderedDict

import supervisely as sly
from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d
from logging import Logger
import os
import open3d
from datasets.utils import BoundingBox, PointCloud
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion


class Tracker3DInterface:
    def __init__(
        self,
        state,
        api,
    ):
        self.api: sly.Api = api
        self.logger: Logger = api.logger
        self.frame_index = state["start_index"]
        self.frames_count = state["settings"]["frames"]

        self.track_id = state["track_id"]
        self.dataset_id = state["dataset_id"]
        self.pc_ids = state["point_cloud_ids"]
        ann = self.api.pointcloud.annotation.download_bulk(
            self.dataset_id, [state["point_cloud_ids"][0]]
        )[0]
        self.figure_ids = [
            fig["id"] for fig in ann["frames"][0]["figures"] if fig["id"] in state["figures_ids"]
        ]
        self.object_ids = [
            fig["objectId"]
            for fig in ann["frames"][0]["figures"]
            if fig["id"] in state["figures_ids"]
        ]
        self.direction = state["settings"]["direction"]

        self.stop = (len(self.figure_ids) * self.frames_count) + (2 * self.frames_count) + 1
        self.global_pos = 0
        self.global_stop_indicatior = False
        self.pc_dir = "./pointclouds/"

        self.geometries = []
        self.frames_indexes = []

        self.add_frames_indexes()
        self.add_geometries()
        self.load_frames()

        self.logger.info("Tracker 3D interface initialized")

    def add_geometries(self):
        for figure_id in self.figure_ids:
            figure = self.api.pointcloud.figure.get_info_by_id(figure_id)
            geometry = sly.deserialize_geometry(figure.geometry_type, figure.geometry)
            self.geometries.append(geometry)

    def add_frames_indexes(self):
        total_frames = len(self.api.pointcloud_episode.get_frame_name_map(self.dataset_id))
        cur_index = self.frame_index

        while 0 <= cur_index < total_frames and len(self.frames_indexes) < self.frames_count:
            self.frames_indexes.append(cur_index)
            cur_index += 1 if self.direction == "forward" else -1

    def preprocess_cuboid(self, geometry):
        assert geometry.geometry_name() == "cuboid_3d"
        # get vectors
        position = geometry.position
        rotation = geometry.rotation
        dimensions = geometry.dimensions

        rot = Rotation.from_rotvec([rotation.x, rotation.y, rotation.z])
        rot_mat = rot.as_matrix()
        center = [
            position.x + dimensions.x / 2,
            position.y + dimensions.y / 2,
            position.z + dimensions.z / 2,
        ]
        size = [dimensions.x, dimensions.y, dimensions.z]
        orientation = Quaternion(matrix=rot_mat)
        return BoundingBox(center, size, orientation)

    def postprocess_cuboid(self, box):
        position = box.center - box.wlh / 2
        position = Vector3d(position[0], position[1], position[2])
        dimensions = Vector3d(box.wlh[0], box.wlh[1], box.wlh[2])
        rot = Rotation.from_quat(box.orientation.elements)
        rot_vec = rot.as_rotvec()
        rotation = Vector3d(rot_vec[0], rot_vec[1], rot_vec[2])
        return Cuboid3d(position, rotation, dimensions)

    def load_frames(self):
        self.frames = []
        self.logger.info(f"Loading {len(self.frames_indexes)} frames...")
        for i, pc_id in enumerate(self.pc_ids):
            frame = {}
            cloud_info = self.api.pointcloud.get_info_by_id(pc_id)
            self.api.pointcloud_episode.download_path(
                cloud_info.id, os.path.join(self.pc_dir, cloud_info.name)
            )
            pcd = open3d.io.read_point_cloud(os.path.join(self.pc_dir, cloud_info.name))
            pcd = PointCloud(np.asarray(pcd.points).T)
            frame["pcd"] = pcd
            if i == 0:
                geometry = self.geometries[0]
                bbox = self.preprocess_cuboid(geometry)
                frame["bbox"] = bbox
            self.frames.append(frame)
            self._notify(task="load frame")

    def _notify(
        self,
        stop: bool = False,
        fstart: Optional[int] = None,
        fend: Optional[int] = None,
        task: str = "not defined",
    ):
        self.global_pos += 1

        if stop:
            pos = self.stop
        else:
            pos = self.global_pos

        fstart = min(self.frames_indexes) if fstart is None else fstart
        fend = max(self.frames_indexes) if fend is None else fend

        self.logger.debug(f"Task: {task}")
        self.logger.debug(f"Notification status: {pos}/{self.stop}")

        # self.global_stop_indicatior = self.api.video.notify_progress(
        #     self.track_id,
        #     self.dataset_id,
        #     fstart,
        #     fend,
        #     pos,
        #     self.stop,
        # )

        self.logger.debug(f"Notification status: stop={self.global_stop_indicatior}")

        if self.global_stop_indicatior and self.global_pos < self.stop:
            self.logger.info("Task stoped by user")

    def add_cuboid_on_frame(self, pcd_id, object_id, cuboid_json, track_id):
        self.api.pointcloud_episode.figure.create(pcd_id, object_id, cuboid_json, "cuboid_3d", track_id)
        self._notify(task="add geometry on frame")
