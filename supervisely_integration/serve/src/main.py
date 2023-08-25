import supervisely as sly
import os
from dotenv import load_dotenv
from typing_extensions import Literal
import yaml
from addict import Dict
from utils import Logger
from tasks import create_task


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
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        model_path = configs_path + "mbptrack_kitti_car_cfg.yaml"
        checkpoint_path = checkpoints_path + "mbptrack_kitti_car.ckpt"
        with open(model_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = Dict(cfg)
        cfg.work_dir = "./work_dir/"
        cfg.resume_from = checkpoint_path
        cfg.save_test_result = True
        cfg.gpus = [0]
        os.makedirs(cfg.work_dir, exist_ok=True)
        with open(os.path.join(cfg.work_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg.to_dict(), f)
        log_file_dir = os.path.join(cfg.work_dir, "3DSOT.log")
        log = Logger(name="3DSOT", log_file=log_file_dir)
        self.task = create_task(cfg, log)

    def predict(
        self,
        frames,
        input_box,
    ):
        # test_dataset = create_datasets(
        #     cfg=cfg.dataset_cfg,
        #     split_types='test',
        #     log=log
        # )
        # test_dataloader = DataLoader(
        #     test_dataset,
        #     shuffle=False,
        #     batch_size=cfg.eval_cfg.batch_size,
        #     pin_memory=False,
        #     num_workers=cfg.eval_cfg.num_workers,
        #     collate_fn=lambda x: x
        # )
        # progress_bar_callback = CustomProgressBar()
        # logger = CustomTensorBoardLogger(
        #     save_dir=cfg.work_dir, version='', name='')

        # trainer = pl.Trainer(
        #     gpus=cfg.gpus,
        #     strategy='ddp',
        #     log_every_n_steps=1,
        #     callbacks=[progress_bar_callback],
        #     default_root_dir=cfg.work_dir,
        #     enable_model_summary=False,
        #     num_sanity_val_steps=0,
        #     logger=logger,
        # )
        # trainer.test(task, test_dataloader, ckpt_path=cfg.resume_from)
        print("Experimental predict")


model = MBPTracker()
model.serve()
#-----for debug-----
# api = sly.Api()
# context = {
#         "trackId": "5b82a928-0566-4d4d-a8e3-35f5abc736fe",
#         "pointcloudId": 72823,
#         "objectIds": [5761741],
#         "figureIds": [107460951],
#         "direction": "forward",
#         "frameIndex": 15,
#         "frames": 5,
#     }
# model.track_debug(context, api)