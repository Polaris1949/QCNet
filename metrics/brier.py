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
from typing import Optional

import torch
from torchmetrics import Metric

from metrics.utils import topk
from metrics.utils import valid_filter



class Brier(Metric):

    def __init__(self,
                 max_guesses: int = 6,    # 表示最大的猜测次数
                 **kwargs) -> None:
        super(Brier, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,      # 标识有效预测的掩码张量
               keep_invalid_final_step: bool = True,           # 是否在最终步骤保留无效数据
               min_criterion: str = 'FDE') -> None:            # 用于确定最佳预测的标准
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)  # 过滤无效数据
        pred_topk, prob_topk = topk(self.max_guesses, pred, prob)       # 用于从预测值和概率值中提取前 max_guesses 个最高概率的预测
        if min_criterion == 'FDE':
            inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
            inds_best = torch.norm(pred_topk[torch.arange(pred.size(0)), :, inds_last] -            # 计算每个预测与目标之间的欧几里得距离,找到距离最小的预测
                                   target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                                   p=2, dim=-1).argmin(dim=-1)
        elif min_criterion == 'ADE':
            inds_best = (torch.norm(pred_topk - target.unsqueeze(1), p=2, dim=-1) *               # 计算所有预测与目标之间的欧几里得距离的加权和，并找到加权和最小的预测
                         valid_mask.unsqueeze(1)).sum(dim=-1).argmin(dim=-1)
        else:
            raise ValueError('{} is not a valid criterion'.format(min_criterion))
        self.sum += (1.0 - prob_topk[torch.arange(pred.size(0)), inds_best]).pow(2).sum()     # 更新 Brier 类的内部状态，将预测概率的平方差累加到 sum 状态，并增加 count 状态的计数
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
