import supervisely as sly
import os
from dotenv import load_dotenv
from typing_extensions import Literal
import yaml
from addict import Dict
from utils import Logger
from models import create_model
import numpy as np
from datasets.utils.pcd_utils import *
import torch
from pytorch_lightning import seed_everything
from datasets.utils import BoundingBox, PointCloud
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d
import open3d

# for debug, has no effect in production
if sly.is_development():
    load_dotenv("supervisely_integration/serve/debug.env")
    load_dotenv("supervisely.env")

configs_path = "./configs/"
checkpoints_path = "/checkpoints/"
checkpoint_name = os.environ.get("modal.state.modelName", "mbptrack_kitti_car.ckpt")


class MBPTracker(sly.nn.inference.ObjectTracking3D):
    def preprocess_cuboid(self, geometry: Cuboid3d):
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

    def postprocess_cuboid(self, box: BoundingBox):
        position = box.center
        position = Vector3d(position[0], position[1], position[2])
        dimensions = Vector3d(box.wlh[0], box.wlh[1], box.wlh[2])
        rot = Rotation.from_matrix(box.rotation_matrix)
        rot_vec = rot.as_rotvec()
        rotation = Vector3d(0, 0, rot_vec[2] - (np.pi / 2))
        return Cuboid3d(position, rotation, dimensions)

    def preprocess_pcd(self, pcd_path):
        pcd = open3d.io.read_point_cloud(pcd_path, format="pcd")
        points = np.asarray(pcd.points, dtype=np.float32)
        pcd = PointCloud(points.T)
        return pcd

    def preprocess_frame(self, frame):
        if "bbox" in frame:
            frame["bbox"] = self.preprocess_cuboid(frame["bbox"])
        frame["pcd"] = self.preprocess_pcd(frame["pcd"])
        return frame

    def preprocess_state_dict(self, state_dict):
        preprocessed_state_dict = {}
        for key, value in state_dict.items():
            preprocessed_state_dict[key[6:]] = value
        return preprocessed_state_dict

    def load_on_device(
        self,
        model_dir: str,
        device: Literal["cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cuda",
    ):
        seed_everything(42)
        sly.logger.debug(f"Checkpoint name: {checkpoint_name}")
        model_path = configs_path + checkpoint_name[:-5] + "_cfg.yaml"
        checkpoint_path = checkpoints_path + checkpoint_name
        with open(model_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.cfg = Dict(cfg)
        self.cfg.work_dir = "./work_dir/"
        os.makedirs(self.cfg.work_dir, exist_ok=True)
        with open(os.path.join(self.cfg.work_dir, "config.yaml"), "w") as f:
            yaml.dump(self.cfg.to_dict(), f)
        log_file_dir = os.path.join(self.cfg.work_dir, "3DSOT.log")
        log = Logger(name="3DSOT", log_file=log_file_dir)
        self.model = create_model(self.cfg.model_cfg, log)
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        raw_state_dict = checkpoint["state_dict"]
        preprocessed_state_dict = self.preprocess_state_dict(raw_state_dict)
        self.model.load_state_dict(preprocessed_state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.memory = None
        self.lwh = None
        self.last_bbox_cpu = np.array([0.0, 0.0, 0.0, 0.0])

    def predict(
        self,
        frame,
        is_last_frame=False,
    ):
        torch.set_grad_enabled(False)
        frame = self.preprocess_frame(frame)
        if "bbox" in frame:
            base_bbox = frame["bbox"]
            self.lwh = np.array([base_bbox.wlh[1], base_bbox.wlh[0], base_bbox.wlh[2]])
        else:
            base_bbox = self.previous_bbox
        pcd = crop_and_center_pcd(
            frame["pcd"],
            base_bbox,
            offset=self.cfg.dataset_cfg.frame_offset,
            offset2=self.cfg.dataset_cfg.frame_offset2,
            scale=self.cfg.dataset_cfg.frame_scale,
        )
        if "bbox" in frame:
            if pcd.nbr_points() == 0:
                pcd.points = np.array([[0.0], [0.0], [0.0]])
            bbox = transform_box(frame["bbox"], base_bbox)
            mask_gt = get_pcd_in_box_mask(pcd, bbox, scale=1.25).astype(int)
            pcd, idx = resample_pcd(
                pcd, self.cfg.dataset_cfg.frame_npts, return_idx=True, is_training=False
            )
            mask_gt = mask_gt[idx]
        else:
            if pcd.nbr_points() <= 1:
                bbox = get_offset_box(
                    self.previous_bbox,
                    self.last_bbox_cpu,
                    use_z=self.cfg.dataset_cfg.eval_cfg.use_z,
                    is_training=False,
                )
                self.previous_bbox = bbox
                return self.postprocess_cuboid(bbox)
            pcd, idx = resample_pcd(
                pcd, self.cfg.dataset_cfg.frame_npts, return_idx=True, is_training=False
            )
        embed_output = self.model(
            dict(
                pcds=torch.tensor(pcd.points.T, device=self.device, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            ),
            mode="embed",
        )
        xyzs, geo_feats, idxs = (
            embed_output["xyzs"],
            embed_output["feats"],
            embed_output["idxs"],
        )

        if "bbox" in frame:
            first_mask_gt = torch.tensor(
                mask_gt, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
            propagate_output = self.model(
                dict(
                    feat=geo_feats[:, 0, :, :],
                    xyz=xyzs[:, 0, :, :],
                    first_mask_gt=torch.gather(first_mask_gt, 1, idxs[:, 0, :]),
                ),
                mode="propagate",
            )
            layer_feats = propagate_output["layer_feats"]
            update_output = self.model(
                dict(
                    layer_feats=layer_feats,
                    xyz=xyzs[:, 0, :, :],
                    mask=torch.gather(first_mask_gt, 1, idxs[:, 0, :]),
                ),
                mode="update",
            )
            self.memory = update_output["memory"]
            self.previous_bbox = frame["bbox"]
            self.pcd_interface._notify(task="cuboid tracking")
            return self.postprocess_cuboid(frame["bbox"])
        else:
            propagate_output = self.model(
                dict(memory=self.memory, feat=geo_feats[:, 0, :, :], xyz=xyzs[:, 0, :, :]),
                mode="propagate",
            )
            geo_feat, mask_feat = propagate_output["geo_feat"], propagate_output["mask_feat"]
            layer_feats = propagate_output["layer_feats"]

            localize_output = self.model(
                dict(
                    geo_feat=geo_feat,
                    mask_feat=mask_feat,
                    xyz=xyzs[:, 0, :, :],
                    lwh=torch.tensor(self.lwh, device=self.device, dtype=torch.float32).unsqueeze(
                        0
                    ),
                ),
                mode="localize",
            )
            mask_pred = localize_output["mask_pred"]
            bboxes_pred = localize_output["bboxes_pred"]
            bboxes_pred_cpu = bboxes_pred.squeeze(0).detach().cpu().numpy()

            bboxes_pred_cpu[np.isnan(bboxes_pred_cpu)] = -1e6

            best_box_idx = bboxes_pred_cpu[:, 4].argmax()
            bbox_cpu = bboxes_pred_cpu[best_box_idx, 0:4]
            if torch.max(mask_pred.sigmoid()) < self.cfg.missing_threshold:
                bbox = get_offset_box(
                    self.previous_bbox,
                    self.last_bbox_cpu,
                    use_z=self.cfg.dataset_cfg.eval_cfg.use_z,
                    is_training=False,
                )
            else:
                bbox = get_offset_box(
                    self.previous_bbox,
                    bbox_cpu,
                    use_z=self.cfg.dataset_cfg.eval_cfg.use_z,
                    is_training=False,
                )
                self.last_bbox_cpu = bbox_cpu

            self.previous_bbox = bbox
            if not is_last_frame:
                update_output = self.model(
                    dict(
                        layer_feats=layer_feats,
                        xyz=xyzs[:, 0, :, :],
                        mask=mask_pred.sigmoid(),
                        memory=self.memory,
                    ),
                    mode="update",
                )
                self.memory = update_output["memory"]
            else:
                self.memory = None
                self.lwh = None
                self.last_bbox_cpu = np.array([0.0, 0.0, 0.0, 0.0])
            self.pcd_interface._notify(task="cuboid tracking")
        return self.postprocess_cuboid(bbox)


model = MBPTracker()
model.serve()
