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


# 处理不完整数据或异常值
class MR(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(MR, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True,
               miss_criterion: str = 'FDE',                 #  缺失标准，用于确定预测是否被认为是缺失的
               miss_threshold: float = 2.0) -> None:        #  缺失阈值，用于确定预测是否被认为是缺失的阈值
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        if miss_criterion == 'FDE':          # 计算每个样本的第一个预测误差超过阈值的情况
            inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
            self.sum += (torch.norm(pred_topk[torch.arange(pred.size(0)), :, inds_last] -     # 通过比较每个样本的最小误差是否大于miss_threshold来确定累加缺失
                                    target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                                    p=2, dim=-1).min(dim=-1)[0] > miss_threshold).sum()
        elif miss_criterion == 'MAXDE':      # 计算每个样本的最大预测误差超过阈值的情况
            self.sum += (((torch.norm(pred_topk - target.unsqueeze(1),          # ：通过比较每个样本的最大误差是否大于miss_threshold来确定累加缺失
                                      p=2, dim=-1) * valid_mask.unsqueeze(1)).max(dim=-1)[0]).min(dim=-1)[0] >
                         miss_threshold).sum()
        else:
            raise ValueError('{} is not a valid criterion'.format(miss_criterion))
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
