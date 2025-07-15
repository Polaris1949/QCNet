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
from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset
from predictors import QCNet


if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)             # 设定随机种子保证实验可重复性
    parser = ArgumentParser()                                # 将命令行参数解析成 Python 数据类型所需的全部信息
    parser.add_argument('--model', type=str, required=True)       # 参数设定  模型
    parser.add_argument('--root', type=str, required=True)        # 根目录
    parser.add_argument('--batch_size', type=int, default=32)     # 测试时批处理大小
    parser.add_argument('--num_workers', type=int, default=8)     # 数据加载时的工作线程数
    parser.add_argument('--pin_memory', type=bool, default=True)  # 是否将数据加载到cuda固定内存中
    parser.add_argument('--persistent_workers', type=bool, default=True)  # 在训练开始后是否保持工程线程
    parser.add_argument('--accelerator', type=str, default='auto')        # 加速器类型
    parser.add_argument('--devices', type=int, default=1)                 # 设备数量
    parser.add_argument('--ckpt_path', type=str, required=True)           # 模型检查点的路径
    args = parser.parse_args()                                # 解析参数

    model = {                                                 # 根据指定的模型名称加载模型
        'QCNet': QCNet,
    }[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path)
    test_dataset = {                                          # 根据模型使用的特定数据集创建测试数据集
        'argoverse_v2': ArgoverseV2Dataset,
    }[model.dataset](root=args.root, split='test')
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,    # 数据加载器：在测试时加载数据
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='dp')              #  管理训练过程  strategy:  'ddp'，在多个 GPU 上并行训练模型的方法，可以加速训练过程
    trainer.test(model, dataloader)                                                                       #  开始测试
