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
from torchmetrics import Metric



class AverageMeter(Metric):

    def __init__(self, **kwargs) -> None:
        super(AverageMeter, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')  # 'sum' 用于累积所有传入值的总和
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')  # 'count' 用于累积传入值的数量

    def update(self, val: torch.Tensor) -> None:
        self.sum += val.sum()           # 将传入的张量 val 的元素求和，并累加到 'sum' 状态
        self.count += val.numel()       # 将传入的张量 val 的元素数量求和，累加到 'count' 状态

    def compute(self) -> torch.Tensor:
        return self.sum / self.count          # 计算所有传入值的平均值
