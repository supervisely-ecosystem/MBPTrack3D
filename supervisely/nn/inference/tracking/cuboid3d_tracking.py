import numpy as np
import functools
from fastapi import Request, BackgroundTasks
from typing import Any, Dict, List, Optional, Union
import supervisely as sly
from supervisely.nn.inference.tracking.tracker3d_interface import Tracker3DInterface
from supervisely.nn.inference import Inference
import os


class Cuboid3DTracking(Inference):
    def __init__(
        self,
        model_dir: Optional[str] = None,
        custom_inference_settings: Optional[Union[Dict[str, Any], str]] = None,
    ):
        super().__init__(
            model_dir,
            custom_inference_settings,
            sliding_window_mode=None,
            use_gui=False,
        )
        self.load_on_device(model_dir, "cuda")

    def get_info(self):
        info = super().get_info()
        info["task type"] = "tracking"
        return info

    def serve(self):
        super().serve()
        server = self._app.get_server()

        @server.post("/interpolate_figures_ids")
        def start_track(request: Request):
            track(request)
            return {"message": "Track task started."}

        def send_error_data(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                value = None
                try:
                    value = func(*args, **kwargs)
                except Exception as exc:
                    request: Request = args[0]
                    state = request.state.state
                    api: sly.Api = request.state.api
                    track_id = state["track_id"]
                    api.logger.error("An error occured:")
                    api.logger.exception(exc)

                    # api.post(
                    #     "videos.notify-annotation-tool",
                    #     data={
                    #         "type": "videos:tracking-error",
                    #         "data": {
                    #             "trackId": track_id,
                    #             "error": {"message": repr(exc)},
                    #         },
                    #     },
                    # )
                return value

            return wrapper

        @send_error_data
        def track(request: Request = None):
            # initialize tracker 3d interface
            state = request.state.state
            api: sly.Api = request.state.api
            self.pcd_interface = Tracker3DInterface(
                state=state,
                api=api,
            )
            api.logger.info("Starting tracking process")
            # propagate frame-by-frame
            for i, pc_id in enumerate(self.pcd_interface.pc_ids):
                # download input data
                frame = {}
                cloud_info = api.pointcloud.get_info_by_id(pc_id)
                pcd_path = os.path.join(self.pcd_interface.pc_dir, cloud_info.name)
                api.pointcloud_episode.download_path(cloud_info.id, pcd_path)
                frame["pcd"] = pcd_path
                if i == 0:
                    geometry = self.pcd_interface.geometries[0]
                    frame["bbox"] = geometry
                self.pcd_interface._notify(task="load frame")
                # pass input data to model and get prediction
                if i == len(self.pcd_interface.pc_ids) - 1:
                    predicted_cuboid = self.predict(frame, is_last_frame=True)
                else:
                    predicted_cuboid = self.predict(frame)
                # add predicted cuboid to frame
                if i != 0:
                    pcd_id = self.pcd_interface.pc_ids[i]
                    obj_id = self.pcd_interface.object_ids[0]
                    track_id = self.pcd_interface.track_id
                    self.pcd_interface.add_cuboid_on_frame(
                        pcd_id, obj_id, predicted_cuboid.to_json(), track_id
                    )
                # clean directory with downloaded pointclouds
                sly.fs.clean_dir(self.pcd_interface.pc_dir)
                api.logger.info("Successfully finished tracking process")
