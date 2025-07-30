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
import os.path
from itertools import chain
from itertools import compress
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from losses import MixtureNLLLoss
from losses import NLLLoss
from metrics import Brier
from metrics import MR
from metrics import minADE
from metrics import minAHE
from metrics import minFDE
from metrics import minFHE
from modules import QCNetDecoder
from modules import QCNetEncoder
from modules.qcnet_agent_encoder import USE_NATSUMI
from utils.mukuro import check_nan


try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object


class QCNet(pl.LightningModule):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps: int,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_dec_layers: int,        # 解码器中的层数
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 lr: float,                   # 学习率
                 weight_decay: float,         # 权重衰减，用于正则化
                 T_max: int,                  # 表示预测的最大时间步
                 submission_dir: str,         # 提交文件的目录
                 submission_file_name: str,   # 提交文件的名称
                 **kwargs) -> None:
        # FIXME: RuntimeError: It looks like your LightningModule has parameters that were not used in producing the loss returned by training_step.
        super(QCNet, self).__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_freq_bands = num_freq_bands
        self.num_map_layers = num_map_layers
        self.num_agent_layers = num_agent_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.pl2pl_radius = pl2pl_radius
        self.time_span = time_span
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_t2m_steps = num_t2m_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.submission_dir = submission_dir
        self.submission_file_name = submission_file_name

        self.encoder = QCNetEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_map_layers=num_map_layers,
            num_agent_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.decoder = QCNetDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            output_head=output_head,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            num_modes=num_modes,
            num_recurrent_steps=num_recurrent_steps,
            num_t2m_steps=num_t2m_steps,
            pl2m_radius=pl2m_radius,
            a2m_radius=a2m_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_dec_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        self.reg_loss = NLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head, # 初始化一个负对数似然损失对象，用于计算模型预测的连续输出的损失
                                reduction='none')   # reduction='none'表示不进行损失的聚合，返回每个样本的损失
        self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head, # 初始化一个混合负对数似然损失对象，用于计算模型预测的离散输出的损失
                                       reduction='none')    # ，根据output_dim和output_head使用不同的分布

        self.Brier = Brier(max_guesses=6)  # 初始化一个Brier分数计算对象，用于评估预测的不确定性和准确性
        self.minADE = minADE(max_guesses=6)  # 初始化一个平均位移误差最小值计算对象，用于评估预测轨迹与真实轨迹之间的接近程度
        self.minAHE = minAHE(max_guesses=6)  # 初始化一个平均角度误差最小值计算对象，用于评估预测头部方向与真实头部方向之间的接近程度
        self.minFDE = minFDE(max_guesses=6)  # 初始化一个最终位移误差最小值计算对象，用于评估预测轨迹在最后一个时间步与真实轨迹之间的接近程度
        self.minFHE = minFHE(max_guesses=6)  # 初始化一个最终角度误差最小值计算对象，用于评估预测头部方向在最后一个时间步与真实头部方向之间的接近程度
        self.MR = MR(max_guesses=6)          # 初始化一个漏检率计算对象，用于评估模型在检测任务中的性能

        self.test_predictions = dict()   # 初始化一个字典，用于存储测试过程中的预测结果

# 前向传播过程
    def forward(self, data: HeteroData):
        scene_enc = self.encoder(data)       # 调用了当前实例的encoder方法，并将data作为参数传递
        pred = self.decoder(data, scene_enc)  # 将 data 和 scene_enc 作为参数传递
        return pred

    # FIXED: DDP boom, FUCKing GRLC, see grlc.py
    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)

# 训练步骤函数，用于处理序列预测问题
    def training_step(self,
                      data,
                      batch_idx):
        if isinstance(data, Batch):      # 检查 data 是否是 Batch 类的实例
            data['agent']['av_index'] += data['agent']['ptr'][:-1]    # 更新data中智能体的v_index，用于跟踪智能体在序列中的位置
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]   # 提取的预测掩码，分别用于回归损失和分类损失的计算
        cls_mask = data['agent']['predict_mask'][:, -1]
        pred = self(data)
        if self.output_head: # 预测结果包括位置、方向和尺度等信息
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else: # 只包括位置和尺度信息
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']       # pi是预测的概率，用于分类损失的计算
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1) # gt是真实的目标数据，用于计算损失
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1) #  计算预测轨迹和真实轨迹之间的L2范数损失
        best_mode = l2_norm.argmin(dim=-1)    # 是损失最小的模式索引，用于选择最佳预测轨迹
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        reg_loss_propose = self.reg_loss(traj_propose_best,          # reg_loss_propose 和 reg_loss_refine分别是提出的轨迹和细化的轨迹的回归损失
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim + self.output_head],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)   # 分类损失，用于计算代理在最后一个时间步的分类概率
        # 记录了三种损失，并且这些损失会被显示在进度条上
        self.log('train_reg_loss_propose', reg_loss_propose, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_reg_loss_refine', reg_loss_refine, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss', cls_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        if USE_NATSUMI:
            # TODO: Integrate GRLC loss
            loss = reg_loss_propose + reg_loss_refine + cls_loss + self.encoder.agent_encoder.natsumi.loss
        else:
            loss = reg_loss_propose + reg_loss_refine + cls_loss  # 总损失loss是回归损失和分类损失的和，这个损失将被用于模型的反向传播

        check_nan(loss, "Training loss contains NaN values")  # 检查损失是否包含NaN值，如果包含则抛出异常

        print(f'the training loss is {loss.item()}')  # 打印当前的训练损失值
        return loss

# 在模型验证阶段计算和记录损失
    def validation_step(self,
                        data,
                        batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        pred = self(data)  # 调用当前实例的forward方法，传入数据data，得到预测结果pred
        # pred是一个字典，包含了模型的预测结果，包括位置、方向、尺度等信息
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim + self.output_head],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        # 函数没有返回值，但是会记录损失，这些损失可以在验证过程中监控模型的性能
        self.log('val_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_cls_loss', cls_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] == 3 # 创建一个 eval_mask，，用于从数据中选择类别3的智能体
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        valid_mask_eval = reg_mask[eval_mask]        # 只在特定类别的智能体上进行回归损失的计算
        traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]   # 从traj_refine中选择出轨迹，只包括eval_mask为真的那些智能体的轨迹
        if not self.output_head:
            traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                                                     traj_eval[..., :2]], dim=-2)   # 在每个轨迹序列的开始处添加了一个零位置，然后与原始的二维轨迹拼接
            motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]  # 计算traj_2d_with_start_pos_eval中连续位置之间的差异得到的，它表示智能体的运动向量
            head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])   # 通过计算motion_vector_eval的反正切函数得到的，它表示智能体的移动方向
            traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)    # 更新为包括原始轨迹和新计算的头部信息
        pi_eval = F.softmax(pi[eval_mask], dim=-1)       # 应用eval_mask到pi上，并对该结果应用softmax函数，得到每个类别的概率分布
        gt_eval = gt[eval_mask]              # 从gt中选择出的真值轨迹，只包括eval_mask为真的那些智能体的轨迹

# 更新和记录多个评估指标的部分
        self.Brier.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                          valid_mask=valid_mask_eval)
        self.minADE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval, # 平均位移误差
                           valid_mask=valid_mask_eval)
        self.minAHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)                   # 平均角度误差
        self.minFDE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval, # 最终位移误差
                           valid_mask=valid_mask_eval)
        self.minFHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)                   # 最终角度误差
        self.MR.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                       valid_mask=valid_mask_eval)
        # 记录指标的值
        self.log('val_Brier', self.Brier, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minAHE', self.minAHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFHE', self.minFHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))

    def test_step(self,
                  data,
                  batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        pred = self(data)
        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        #print(f'{traj_refine.shape=}, {pred["pi"].shape=}')  # 打印预测轨迹和概率的形状，用于调试
        pi = pred['pi']
        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] == 3
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]  # origin_eval 和 theta_eval 分别是被选中智能体的历史位置和方向
        theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)        # 旋转矩阵，用于将智能体的预测轨迹从全局坐标系转换到智能体的本地坐标系
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        traj_eval = torch.matmul(traj_refine[eval_mask, :, :, :2],    # 转换后的预测轨迹，它是通过将traj_refine与旋转矩阵相乘并加上原始位置来得到的
                                 rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        pi_eval = F.softmax(pi[eval_mask], dim=-1)      # 应用eval_mask到pi上，并对该结果应用softmax函数，得到每个类别的概率分布

        traj_eval = traj_eval.cpu().numpy()      # 这行代码将traj_eval张量从GPU（如果它在GPU上）转移到CPU，然后将其转换为NumPy数组
        pi_eval = pi_eval.cpu().numpy()          # 将 pi_eval 张量转移到CPU并转换为NumPy数组
        if self.dataset == 'argoverse_v2':
            eval_id = list(compress(list(chain(*data['agent']['id'])), eval_mask))   # 使用eval_mask来选择特定类别的智能体ID。chain将所有智能体的ID拼接成一个长列表，compress函数根据eval_mask选择相应的ID
            if isinstance(data, Batch):
                for i in range(data.num_graphs):  # 码遍历每个图（或场景）
                    self.test_predictions[data['scenario_id'][i]] = (pi_eval[i], {eval_id[i]: traj_eval[i]})   # 将预测的概率和轨迹存储在self.test_predictions字典中，其中键是场景ID，值是一个元组，包含概率和轨迹的字典
            else:
                self.test_predictions[data['scenario_id']] = (pi_eval[0], {eval_id[0]: traj_eval[0]})      # 将预测的概率和轨迹存储在self.test_predictions字典中，单个场景的情况
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        # TODO
        #print(f'{pi.shape=}, {pi_eval.shape=}')
        self.save_test_parquet(save_dir=self.submission_dir)  # 保存测试预测结果到parquet文件中，使用self.submission_dir作为保存目录
        #exit(0)

    def on_test_end(self):
        if self.dataset == 'argoverse_v2':
            ChallengeSubmission(self.test_predictions).to_parquet(
                Path(self.submission_dir) / f'{self.submission_file_name}.parquet')
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def save_test_parquet(self, save_dir: str):
        """
        Save the test predictions to a parquet file, same as the input dataset format.
        :param save_dir: The directory to save the parquet files.
        """
        from av2.datasets.motion_forecasting.scenario_serialization import (
            ArgoverseScenario, Track, ObjectState, serialize_argoverse_scenario_parquet, load_argoverse_scenario_parquet
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for scenario_id, (pi, traj) in self.test_predictions.items():
            scenario = load_argoverse_scenario_parquet(Path(f'./data_av2/test/raw/{scenario_id}/scenario_{scenario_id}.parquet'))  # 从保存的parquet文件中加载场景数据
            # scenario_id is the key, pi is the predicted probabilities, traj is the predicted trajectories
            # traj is a dict with agent id as key and trajectory as value
            for agent_id, trajectory in traj.items():  # 遍历每个智能体的预测轨迹
                # Find target track in scenario
                target_track = next((track for track in scenario.tracks if track.track_id == agent_id), None)
                if target_track is None:  # 如果没有找到目标轨迹，则跳过
                    continue
                #print(f'{target_track.object_states=}')
                # trajectory.shape=(6, 60, 2) [num_modes, num_future_steps, output_dim]
                trajectory = trajectory[0]
                object_states = target_track.object_states  # 创建一个空的对象状态列表，用于存储智能体的状态
                if len(object_states) < self.num_historical_steps + self.num_future_steps:  # 如果对象状态的长度小于历史步数和未来步数之和，则跳过
                    continue
                #print(f'{target_track=}, {len(object_states)=}, {self.num_historical_steps=}, {self.num_future_steps=}')  # 打印对象状态的长度和历史步数、未来步数
                for t in range(self.num_future_steps):  # 遍历每个时间步
                    pos_x = trajectory[t, 0]  # 获取当前时间步的x坐标
                    pos_y = trajectory[t, 1]  # 获取当前时间步的y坐标
                    object_state = object_states[t + self.num_historical_steps]  # 获取当前时间步对应的对象状态
                    object_state.observed = True
                    object_state.position = (pos_x, pos_y)  # 更新对象状态的位置信息s
            serialize_argoverse_scenario_parquet(Path(f'./test_output/{scenario_id}.parquet'), scenario)

    # 配置模型的优化器
    def configure_optimizers(self):
        decay = set()       # 分别用于存储应该应用权重衰减和不应用权重衰减
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,  # 分别包含了应该应用权重衰减和不应该应用权重衰减
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():  # 循环遍历模型中的所有模块
            # TODO: Integrate GRLC optimizers
            # if module_name.startswith('encoder.agent_encoder.natsumi'):
            #     continue
            # print(f'{module_name=}')
            for param_name, param in module.named_parameters():     # 循环遍历模中块的所有参数
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name      #创建一个包含模块名称和参数名称的完整参数名称
                if 'bias' in param_name:
                    no_decay.add(full_param_name)       # 不应用权重衰减，将其添加到no_decay集合中
                elif 'weight' in param_name:            # 根据模块的类型决定是否应用权重衰减
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)              # 不应用权重衰减，将其添加到no_decay集合中
        param_dict = {param_name: param for param_name, param in self.named_parameters()}       # 创建一个字典，将参数名称映射到参数对象
        inter_params = decay & no_decay                       # 分别计算两个集合的交集和并集
        union_params = decay | no_decay
        assert len(inter_params) == 0                         # 确保没有参数同时存在于decay和no_decay集合中
        assert len(param_dict.keys() - union_params) == 0     # 确保所有参数都被正确分类到decay或no_decay集合中

        # # QCNet的特征表征层设置为需要梯度,GRLC的参数设置为需要梯度
        # parameters_to_update = [
        #     "encoder.agent_encoder.type_a_emb.weight",
        #     "encoder.agent_encoder.x_a_emb.freqs.weight",
        #     "encoder.agent_encoder.x_a_emb.mlps.0.0.weight",
        #     "encoder.agent_encoder.x_a_emb.mlps.0.1.weight",
        #     "encoder.agent_encoder.x_a_emb.mlps.0.3.weight",
        #     "encoder.agent_encoder.x_a_emb.mlps.1.0.weight",
        #     "encoder.agent_encoder.x_a_emb.mlps.1.1.weight",
        #     "encoder.agent_encoder.x_a_emb.mlps.1.3.weight",
        #     "encoder.agent_encoder.x_a_emb.mlps.2.0.weight",
        #     "encoder.agent_encoder.x_a_emb.mlps.2.1.weight",
        #     "encoder.agent_encoder.x_a_emb.mlps.2.3.weight",
        #     "encoder.agent_encoder.x_a_emb.mlps.3.0.weight",
        #     "encoder.agent_encoder.x_a_emb.mlps.3.1.weight",
        #     "encoder.agent_encoder.x_a_emb.mlps.3.3.weight",
        #     "encoder.agent_encoder.x_a_emb.to_out.0.weight",
        #     "encoder.agent_encoder.x_a_emb.to_out.2.weight",
        #     "encoder.agent_encoder.natsumi.model.gcn_0.fc.weight",
        #     "encoder.agent_encoder.natsumi.model.gcn_1.fc.weight",
        #
        #     "encoder.agent_encoder.x_a_emb.mlps.0.0.bias",
        #     "encoder.agent_encoder.x_a_emb.mlps.0.1.bias",
        #     "encoder.agent_encoder.x_a_emb.mlps.0.3.bias",
        #     "encoder.agent_encoder.x_a_emb.mlps.1.0.bias",
        #     "encoder.agent_encoder.x_a_emb.mlps.1.1.bias",
        #     "encoder.agent_encoder.x_a_emb.mlps.1.3.bias",
        #     "encoder.agent_encoder.x_a_emb.mlps.2.0.bias",
        #     "encoder.agent_encoder.x_a_emb.mlps.2.1.bias",
        #     "encoder.agent_encoder.x_a_emb.mlps.2.3.bias",
        #     "encoder.agent_encoder.x_a_emb.mlps.3.0.bias",
        #     "encoder.agent_encoder.x_a_emb.mlps.3.1.bias",
        #     "encoder.agent_encoder.x_a_emb.mlps.3.3.bias",
        #     "encoder.agent_encoder.x_a_emb.to_out.0.bias",
        #     "encoder.agent_encoder.x_a_emb.to_out.2.bias",
        #     "encoder.agent_encoder.natsumi.model.gcn_0.fc.bias",
        #     "encoder.agent_encoder.natsumi.model.gcn_1.fc.bias",
        #     "encoder.agent_encoder.natsumi.model.gcn_1.bias",
        #     "encoder.agent_encoder.natsumi.model.gcn_0.bias"
        # ]
        #
        # # Freeze parameters that are not related to GRLC
        # for param_name in sorted(list(param_dict.keys())):  # 遍历所有参数
        #     #print(f"{param_name}: 设置前：requires_grad={param_dict[param_name].requires_grad}")
        #         if param_name in parameters_to_update:
        #             param_dict[param_name].requires_grad = True
        #         else:
        #             param_dict[param_name].requires_grad = False  # TODO: 将其他参数的requires_grad属性设置为False，表示这些参数在训练过程中不需要计算梯度


        optim_groups = [     # 将参数传递给优化器
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],  # params键对应的值是一个列表，包含了所有应该应用权重衰减的参数
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        for param_name in sorted(list(param_dict.keys())):  # 遍历所有参数
            print(f"{param_name}: 设置后：requires_grad={param_dict[param_name].requires_grad}")  # 打印每个参数的名称和其requires_grad属性的值


        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)    # 创建了一个AdamW优化器实例，正确地应用权重衰减。这里，lr是学习率，weight_decay是全局设置的权重衰减值
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)  # 创建了一个学习率调度器，它将学习率按照余弦衰减函数进行调整。T_max是一个周期内的学习率调度次数，eta_min是学习率衰减到的最小值
        return [optimizer], [scheduler]      # 返回一个包含优化器的列表和一个包含学习率调度器的列表

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('QCNet')
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--output_head', action='store_true')
        parser.add_argument('--num_historical_steps', type=int, required=True)
        parser.add_argument('--num_future_steps', type=int, required=True)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--num_recurrent_steps', type=int, required=True)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_map_layers', type=int, default=1)
        parser.add_argument('--num_agent_layers', type=int, default=2)
        parser.add_argument('--num_dec_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)       # Dropout比率，默认为0.1
        parser.add_argument('--pl2pl_radius', type=float, required=True)
        parser.add_argument('--time_span', type=int, default=None)
        parser.add_argument('--pl2a_radius', type=float, required=True)
        parser.add_argument('--a2a_radius', type=float, required=True)
        parser.add_argument('--num_t2m_steps', type=int, default=None)          # 时间到模型的步数，默认为None
        parser.add_argument('--pl2m_radius', type=float, required=True)
        parser.add_argument('--a2m_radius', type=float, required=True)
        parser.add_argument('--lr', type=float, default=5e-4)         # 学习率
        parser.add_argument('--weight_decay', type=float, default=1e-4) # 权重衰减
        parser.add_argument('--T_max', type=int, default=64)          # 最大周期
        parser.add_argument('--submission_dir', type=str, default='./')  # 提交目录，默认为当前目录
        parser.add_argument('--submission_file_name', type=str, default='submission')   # 提交文件名，默认为 'submission'
        return parent_parser
