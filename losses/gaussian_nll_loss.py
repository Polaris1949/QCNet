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
import torch.nn.functional as F


class GaussianNLLLoss(nn.Module):

    def __init__(self,
                 full: bool = False,
                 eps: float = 1e-6,                      # 用于数值稳定性，防止对数运算时出现数值问题
                 reduction: str = 'mean') -> None:       # 损失的缩减方式,计算所有样本损失的平均值
        super(GaussianNLLLoss, self).__init__()
        self.full = full
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        mean, var = pred.chunk(2, dim=-1)         # 将预测值 pred 在最后一个维度上分割成两部分，第一部分是均值 mean，第二部分是方差 var
        return F.gaussian_nll_loss(input=mean, target=target, var=var, full=self.full, eps=self.eps,
                                   reduction=self.reduction)                # 计算损失，返回结果
