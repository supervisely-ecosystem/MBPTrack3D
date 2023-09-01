import numpy as np
from typing import Generator, Optional, List, Tuple, OrderedDict, Dict
from collections import OrderedDict

import supervisely as sly
from supervisely.geometry.geometry import Geometry
from logging import Logger


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
        ann = self.api.pointcloud.annotation.download_bulk(self.dataset_id, [state["point_cloud_ids"][0]])[0]
        self.figure_ids = [fig["id"] for fig in ann["frames"][0]["figures"] if fig["id"] in state["figures_ids"]]
        self.object_ids = [fig["objectId"] for fig in ann["frames"][0]["figures"] if fig["id"] in state["figures_ids"]]
        self.direction = state["settings"]["direction"]

        self.stop = (len(self.figure_ids) * self.frames_count) + self.frames_count + 1
        self.global_pos = 0
        self.global_stop_indicatior = False

        self.geometries = []
        self.frames_indexes = []

        self.add_frames_indexes()
        self.add_geometries()

        self.logger.info("Tracker 3D interface initialized")

    def add_geometries(self):
        for figure_id in self.figure_ids:
            figure = self.api.pointcloud.figure.get_info_by_id(figure_id)
            geometry = sly.deserialize_geometry(figure.geometry_type, figure.geometry)
            self.geometries.append(geometry)
            self._notify(task="add geometry on frame")

    def add_frames_indexes(self):
        total_frames = len(self.api.pointcloud_episode.get_frame_name_map(self.dataset_id))
        cur_index = self.frame_index

        while 0 <= cur_index < total_frames and len(self.frames_indexes) < self.frames_count + 1:
            self.frames_indexes.append(cur_index)
            cur_index += (1 if self.direction == 'forward' else -1)

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
            self.logger.info("Task stoped by user.")