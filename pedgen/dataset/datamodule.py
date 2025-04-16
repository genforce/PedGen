"""Lightning wrapper of the pytorch datamodule."""
from typing import Dict, List, Optional

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from pedgen.dataset.carla_dataset import CarlaDataset
from pedgen.dataset.citywalkers_dataset import (CityWalkersDataset,
                                                collate_fn_pedmotion,
                                                collate_fn_pedmotion_pred)
from pedgen.dataset.sloper4d_dataset import SLOPER4D
from pedgen.dataset.waymo_dataset import WaymoDataset, collate_fn_waymo


class PedGenDataModule(LightningDataModule):
    """Lightning datamodule for pedestrian generation."""

    def __init__(self,
                 train_label_file: str,
                 val_label_file: Optional[str],
                 test_label_file: Optional[str],
                 pred_label_file: Optional[str],
                 batch_size_per_device: int,
                 num_workers: int,
                 data_root: str,
                 img_root: str,
                 img_dim: List,
                 num_timestamp: int,
                 min_timestamp: int,
                 use_partial: bool,
                 depth_root: str = "depth",
                 semantic_root: str = "semantic",
                 use_data_augmentation: bool = False,
                 sample_interval: int = 30,
                 sample_start_idx: int = 0,
                 grid_size: List = [-4, 4, -2, 2, -4, 4],
                 grid_points: List = [40, 40, 40],
                 train_sloper4d: bool = False,
                 use_image: bool = False,
                 test_carla: bool = False,
                 test_waymo: bool = False,
                 train_percent: float = 1.0,
                 carla_conf: Optional[Dict] = None) -> None:
        super().__init__()
        self.train_label_file = train_label_file
        if val_label_file is None:
            self.val_label_file = self.train_label_file
        else:
            self.val_label_file = val_label_file
        if test_label_file is None:
            self.test_label_file = self.val_label_file
        else:
            self.test_label_file = test_label_file
        if pred_label_file is None:
            self.pred_label_file = self.test_label_file
        else:
            self.pred_label_file = pred_label_file
        self.batch_size_per_device = batch_size_per_device
        self.num_workers = num_workers
        self.dataset_conf = {
            "data_root": data_root,
            "img_root": img_root,
            "img_dim": img_dim,
            "min_timestamp": min_timestamp,
            "use_partial": use_partial,
            "num_timestamp": num_timestamp,
            "depth_root": depth_root,
            "sample_interval": sample_interval,
            "sample_start_idx": sample_start_idx,
            "semantic_root": semantic_root,
            "grid_size": grid_size,
            "grid_points": grid_points,
            "train_percent": train_percent,
            "use_data_augmentation": use_data_augmentation,
            "use_image": use_image,
        }
        self.test_waymo = test_waymo
        self.train_sloper4d = train_sloper4d
        self.test_carla = test_carla
        self.carla_conf = carla_conf

    def setup(self, stage: str) -> None:
        if self.test_waymo:
            self.val = WaymoDataset(**self.dataset_conf)
            self.collate_fn_val = collate_fn_waymo
            return

        if not self.train_sloper4d:
            self.train = CityWalkersDataset(label_file=self.train_label_file,
                                            mode="train",
                                            **self.dataset_conf)
            self.val = CityWalkersDataset(label_file=self.val_label_file,
                                          mode="val",
                                          **self.dataset_conf)

        else:
            self.train = SLOPER4D(mode="train", **self.dataset_conf)
            self.val = SLOPER4D(mode="val", **self.dataset_conf)

        if self.test_carla:
            assert self.carla_conf
            self.test = CarlaDataset(
                mode="test",
                img_dim=self.dataset_conf["img_dim"],
                grid_size=self.dataset_conf["grid_size"],
                grid_points=self.dataset_conf["grid_points"],
                **self.carla_conf)
            self.pred = CarlaDataset(
                mode="pred",
                img_dim=self.dataset_conf["img_dim"],
                grid_size=self.dataset_conf["grid_size"],
                grid_points=self.dataset_conf["grid_points"],
                **self.carla_conf)
        else:
            self.test = CityWalkersDataset(label_file=self.test_label_file,
                                           mode="test",
                                           **self.dataset_conf)
            self.pred = CityWalkersDataset(label_file=self.pred_label_file,
                                           mode="pred",
                                           **self.dataset_conf)
        self.collate_fn_train = collate_fn_pedmotion
        self.collate_fn_val = collate_fn_pedmotion
        self.collate_fn_test = collate_fn_pedmotion
        self.collate_fn_pred = collate_fn_pedmotion_pred

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train,
                          shuffle=True,
                          drop_last=True,
                          batch_size=self.batch_size_per_device,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn_train)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val,
                          shuffle=False,
                          drop_last=False,
                          batch_size=self.batch_size_per_device,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn_val)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test,
                          shuffle=False,
                          drop_last=False,
                          batch_size=self.batch_size_per_device,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn_test)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.pred,
                          shuffle=False,
                          drop_last=False,
                          batch_size=self.batch_size_per_device,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn_pred)
