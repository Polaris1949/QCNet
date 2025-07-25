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
import math
from typing import List, Optional

import torch
import torch.nn as nn

from utils import weight_init



class FourierEmbedding(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_freq_bands: int) -> None:              # 表示频率带的数量，这可能用于控制嵌入过程中的频率变化
        super(FourierEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

# 初始化嵌入层、多层感知机（MLPs）、层归一化（LayerNorm）和线性层
        self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None
        self.mlps = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
                for _ in range(input_dim)])
        self.to_out = nn.Sequential(                           # nn.Sequential 模块，用于在最后将MLPs的输出转换回hidden_dim维度
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.apply(weight_init)

    def forward(self,
                continuous_inputs: Optional[torch.Tensor] = None,
                categorical_embs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        if continuous_inputs is None:
            if categorical_embs is not None:
                x = torch.stack(categorical_embs).sum(dim=0)            # 将所有分类型嵌入堆叠起来，并沿第一个维度求和，得到一个张量
            else:
                raise ValueError('Both continuous_inputs and categorical_embs are None')
        else:
            x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
            # Warning: if your data are noisy, don't use learnable sinusoidal embedding
            x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
            continuous_embs: List[Optional[torch.Tensor]] = [None] * self.input_dim
            for i in range(self.input_dim):
                continuous_embs[i] = self.mlps[i](x[:, i])        # 使用多层感知，处理每个维度的连续型输入
            x = torch.stack(continuous_embs).sum(dim=0)
            if categorical_embs is not None:
                x = x + torch.stack(categorical_embs).sum(dim=0)           # 将连续型和分类型嵌入相加。
        return self.to_out(x)
