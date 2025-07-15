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


class LaplaceNLLLoss(nn.Module):

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        with torch.no_grad():              # 暂时禁用梯度计算
            scale.clamp_(min=self.eps)     #  将 scale 的值限制在一个最小值 eps 以上，以避免数值不稳定
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        if self.reduction == 'mean':         # 检查 reduction 参数是否为 'mean'，如果是，则计算损失的平均值。
            return nll.mean()
        elif self.reduction == 'sum':       # 检查 reduction 参数是否为 'sum'，如果是，则计算损失的总和。
            return nll.sum()
        elif self.reduction == 'none':       # 检查 reduction 参数是否为 'none'，如果是，则返回每个样本的损失。
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
