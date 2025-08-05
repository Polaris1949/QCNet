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
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datamodules import ArgoverseV2DataModule
from predictors import QCNet


import warnings

# 忽略UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)          # 是否打乱数据
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=64)           # 最大训练周期
    parser.add_argument('--ckpt_path', type=str, default=None)    # 模型检查点的路径
    parser.add_argument('--natsumi', action='store_true')  # 是否使用Natsumi模型
    parser.add_argument('--natsumi_ckpt', type=str, default=None)  # Natsumi预训练模型的路径
    parser.add_argument('--natsumi_freeze', action='store_true')  # 是否冻结Natsumi模型的参数
    parser.add_argument('--natsumi_feat_qcnet', action='store_true')  # 是否使用QCNet的特征作为Natsumi输入
    parser.add_argument('--num_grlc_steps', type=int, default=10)  # GRLC步骤数
    parser.add_argument('--save_grlc_structure', action='store_true')  # 是否保存GRLC结构
    QCNet.add_model_specific_args(parser)          # 允许模型添加特定的参数
    args = parser.parse_args()

    if args.ckpt_path is None:
        model = QCNet(**vars(args))
    else:
        model = QCNet.load_from_checkpoint(checkpoint_path=args.ckpt_path)

    datamodule = {
        'argoverse_v2': ArgoverseV2DataModule,
    }[args.dataset](**vars(args))
    model_checkpoint = ModelCheckpoint(monitor='val_minFDE', save_top_k=5, mode='min')   # model_checkpoint 是一个模型检查点回调；最小最终位移误差minFDE，保存最佳5个模型
    lr_monitor = LearningRateMonitor(logging_interval='epoch')                  # 学习率监控器，用于在每个epoch后记录学习率
    find_unused_parameters = args.natsumi_ckpt is not None and args.natsumi_freeze is True  # 是否冻结Natsumi模型的参数，若冻结则在分布式训练中查找未使用的参数
    strategy = DDPStrategy(process_group_backend='gloo', find_unused_parameters=find_unused_parameters, gradient_as_bucket_view=True)  # DDPStrategy用于在多个 GPU 上并行训练模型；find_unused_parameters=False 参数用于优化性能，当模型非常大且包含未使用的参数时，可以提高效率；gradient_as_bucket_view=True 是一个性能优化选项，它允许在多 GPU 训练时减少内存使用。
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,    # trainer包含了训练、验证和测试过程的所有细节
                         strategy=strategy,  # strategy: 指定分布式训练的策略
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs)  #callbacks: 指定在训练过程中要使用的回调函数列表
    trainer.fit(model, datamodule)  # 数据预处理，得到整体的指标，训练时使用
