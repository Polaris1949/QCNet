# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset
from transforms import TargetBuilder



class ArgoverseV2DataModule(pl.LightningDataModule):

    def __init__(self,
                 root: str,                                 # 路径
                 train_batch_size: int,
                 val_batch_size: int,
                 test_batch_size: int,
                 shuffle: bool = True,                      # 布尔值，是否在每个epoch开始时对数据进行随机打乱
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_raw_dir: Optional[str] = None,        # 训练集、验证集和测试集的原始数据目录
                 val_raw_dir: Optional[str] = None,
                 test_raw_dir: Optional[str] = None,
                 train_processed_dir: Optional[str] = None,    # 预处理数据目录
                 val_processed_dir: Optional[str] = None,
                 test_processed_dir: Optional[str] = None,
                 train_transform: Optional[Callable] = TargetBuilder(50, 60),       # Callable类型调用  数据的转换操作
                 val_transform: Optional[Callable] = TargetBuilder(50, 60),
                 test_transform: Optional[Callable] = None,          #  None意味着不进行转换
                 **kwargs) -> None:              # 接收其他任意数量的关键字参数
        super(ArgoverseV2DataModule, self).__init__()                # 调用父类pl.LightningDataModule的功能
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

    def prepare_data(self) -> None:               # 一次性的数据准备操作：确保数据集已经被加载和处理
        ArgoverseV2Dataset(self.root, 'train', self.train_raw_dir, self.train_processed_dir, self.train_transform)
        ArgoverseV2Dataset(self.root, 'val', self.val_raw_dir, self.val_processed_dir, self.val_transform)
        ArgoverseV2Dataset(self.root, 'test', self.test_raw_dir, self.test_processed_dir, self.test_transform)

    def setup(self, stage: Optional[str] = None) -> None:      # 为训练、验证和测试阶段准备数据：确保最新的数据集
        self.train_dataset = ArgoverseV2Dataset(self.root, 'train', self.train_raw_dir, self.train_processed_dir,
                                                self.train_transform)
        self.val_dataset = ArgoverseV2Dataset(self.root, 'val', self.val_raw_dir, self.val_processed_dir,
                                              self.val_transform)
        self.test_dataset = ArgoverseV2Dataset(self.root, 'test', self.test_raw_dir, self.test_processed_dir,
                                               self.test_transform)

    def train_dataloader(self):              # 数据加载器：对数据集的批量处理、打乱、多线程加载等
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)
