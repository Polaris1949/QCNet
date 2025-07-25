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
import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


class FocalLoss(nn.Module):

    def __init__(self,
                 alpha: float = 0.25,          # 平衡正负样本的权重
                 gamma: float = 2.0,           # 调节难易样本权重的焦点参数
                 reduction: str = 'mean'):     # 指定损失的缩减方式
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,              # 模型预测的概率
                target: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(pred, target, self.alpha, self.gamma, self.reduction)     # 计算焦点损失
