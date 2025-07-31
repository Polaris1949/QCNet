# Below is dataset generation middleware, Miku.
import pytorch_lightning as pl
from typing import Optional, Callable, Dict, Any, List
from datamodules.miku_datamodule import MikuDataModule
from datasets.yamai_dataset import YamaiDataset
from torch.utils.data import DataLoader as EasyDataLoader

class YamaiDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        train_raw_dir: Optional[str] = None,
        val_raw_dir: Optional[str] = None,
        test_raw_dir: Optional[str] = None,
        train_processed_dir: Optional[str] = None,
        val_processed_dir: Optional[str] = None,
        test_processed_dir: Optional[str] = None,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.train_processed_dir = train_processed_dir
        self.val_processed_dir = val_processed_dir
        self.test_processed_dir = test_processed_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def prepare_data(self) -> None:
        MikuDataModule(self.root, 1, 1, 1).prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = YamaiDataset(self.root, 'train')
        self.val_dataset = YamaiDataset(self.root, 'val')
        self.test_dataset = YamaiDataset(self.root, 'test')

    def train_dataloader(self):
        return EasyDataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return EasyDataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return EasyDataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
