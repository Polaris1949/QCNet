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
from utils import wrap_angle

class minAHE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(minAHE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True,
               min_criterion: str = 'FDE') -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        if min_criterion == 'FDE':
            inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
            inds_best = torch.norm(
                pred_topk[torch.arange(pred.size(0)), :, inds_last, :-1] -
                target[torch.arange(pred.size(0)), inds_last, :-1].unsqueeze(-2), p=2, dim=-1).argmin(dim=-1)
        elif min_criterion == 'ADE':
            inds_best = (torch.norm(pred_topk[..., :-1] - target[..., :-1].unsqueeze(1), p=2, dim=-1) *
                         valid_mask.unsqueeze(1)).sum(dim=-1).argmin(dim=-1)
        else:
            raise ValueError('{} is not a valid criterion'.format(min_criterion))
        # wrap_angle 函数用于处理角度数据，确保角度差异在正确的范围内
        self.sum += ((wrap_angle(pred_topk[torch.arange(pred.size(0)), inds_best, :, -1] - target[..., -1]).abs() *
                      valid_mask).sum(dim=-1) / valid_mask.sum(dim=-1)).sum()                   # 计算预测值和目标值之间的平均角度差异
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
