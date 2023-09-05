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

# for debug, has no effect in production
if sly.is_development():
    load_dotenv("supervisely_integration/serve/debug.env")
    load_dotenv("supervisely.env")

configs_path = "./configs/"
checkpoints_path = "./checkpoints/"


class MBPTracker(sly.nn.inference.Cuboid3DTracking):
    def load_on_device(
        self,
        model_dir: str,
        device: Literal["cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cuda",
    ):
        seed_everything(42)
        model_path = configs_path + "mbptrack_kitti_car_cfg.yaml"
        checkpoint_path = checkpoints_path + "mbptrack_kitti_car.ckpt"
        with open(model_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.cfg = Dict(cfg)
        self.cfg.work_dir = "./work_dir/"
        self.cfg.resume_from = checkpoint_path
        self.cfg.save_test_result = True
        self.cfg.gpus = [0]
        os.makedirs(self.cfg.work_dir, exist_ok=True)
        with open(os.path.join(self.cfg.work_dir, "config.yaml"), "w") as f:
            yaml.dump(self.cfg.to_dict(), f)
        log_file_dir = os.path.join(self.cfg.work_dir, "3DSOT.log")
        log = Logger(name="3DSOT", log_file=log_file_dir)
        self.model = create_model(self.cfg.model_cfg, log)
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        frames,
    ):
        torch.set_grad_enabled(False)
        pred_bboxes = []
        memory = None
        lwh = None
        last_bbox_cpu = np.array([0.0, 0.0, 0.0, 0.0])

        for frame_idx, frame in enumerate(frames):
            if frame_idx == 0:
                base_bbox = frame["bbox"]
                lwh = np.array([base_bbox.wlh[1], base_bbox.wlh[0], base_bbox.wlh[2]])
            else:
                base_bbox = pred_bboxes[-1]
            pcd = crop_and_center_pcd(
                frame["pcd"],
                base_bbox,
                offset=self.cfg.dataset_cfg.frame_offset,
                offset2=self.cfg.dataset_cfg.frame_offset2,
                scale=self.cfg.dataset_cfg.frame_scale,
            )
            if frame_idx == 0:
                if pcd.nbr_points() == 0:
                    pcd.points = np.array([[0.0], [0.0], [0.0]])
                bbox = transform_box(frame["bbox"], base_bbox)
                mask_gt = get_pcd_in_box_mask(pcd, bbox, scale=1.25).astype(int)
                # bbox_gt = np.array([bbox.center[0], bbox.center[1], bbox.center[2], (
                #     bbox.orientation.degrees if self.cfg.dataset_cfg.degree else bbox.orientation.radians) * bbox.orientation.axis[-1]])
                pcd, idx = resample_pcd(
                    pcd, self.cfg.dataset_cfg.frame_npts, return_idx=True, is_training=False
                )
                mask_gt = mask_gt[idx]
            else:
                if pcd.nbr_points() <= 1:
                    bbox = get_offset_box(
                        pred_bboxes[-1],
                        last_bbox_cpu,
                        use_z=self.cfg.dataset_cfg.eval_cfg.use_z,
                        is_training=False,
                    )
                    pred_bboxes.append(bbox)
                    continue
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

            if frame_idx == 0:
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
                memory = update_output["memory"]
                pred_bboxes.append(frame["bbox"])
            else:
                propagate_output = self.model(
                    dict(memory=memory, feat=geo_feats[:, 0, :, :], xyz=xyzs[:, 0, :, :]),
                    mode="propagate",
                )
                geo_feat, mask_feat = propagate_output["geo_feat"], propagate_output["mask_feat"]
                layer_feats = propagate_output["layer_feats"]

                localize_output = self.model(
                    dict(
                        geo_feat=geo_feat,
                        mask_feat=mask_feat,
                        xyz=xyzs[:, 0, :, :],
                        lwh=torch.tensor(lwh, device=self.device, dtype=torch.float32).unsqueeze(0),
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
                        pred_bboxes[-1],
                        last_bbox_cpu,
                        use_z=self.cfg.dataset_cfg.eval_cfg.use_z,
                        is_training=False,
                    )
                else:
                    bbox = get_offset_box(
                        pred_bboxes[-1],
                        bbox_cpu,
                        use_z=self.cfg.dataset_cfg.eval_cfg.use_z,
                        is_training=False,
                    )
                    last_bbox_cpu = bbox_cpu

                pred_bboxes.append(bbox)
                if frame_idx < len(frames) - 1:
                    update_output = self.model(
                        dict(
                            layer_feats=layer_feats,
                            xyz=xyzs[:, 0, :, :],
                            mask=mask_pred.sigmoid(),
                            memory=memory,
                        ),
                        mode="update",
                    )
                    memory = update_output["memory"]
            self.pcd_interface._notify(task="cuboid tracking")
        return pred_bboxes


model = MBPTracker()
model.serve()
