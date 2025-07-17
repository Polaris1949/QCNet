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
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from utils import weight_init



class AttentionLayer(MessagePassing):

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 bipartite: bool,
                 has_pos_emb: bool,
                 **kwargs) -> None:
        super(AttentionLayer, self).__init__(aggr='add', node_dim=0, **kwargs)           # 聚合节点特征采用相加
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.has_pos_emb = has_pos_emb
        self.scale = head_dim ** -0.5          # 计算缩放因子，head_dim的负一半次幂作为缩放因子

# 初始化线性层、Dropout层、前馈多层感知机（MLP），以及层归一化（LayerNorm）
        self.to_q = nn.Linear(hidden_dim, head_dim * num_heads)       # 创建一个线性层，用于将输入特征（维度为 hidden_dim）映射到查询（Query）向量
        self.to_k = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
        self.to_v = nn.Linear(hidden_dim, head_dim * num_heads)
        if has_pos_emb:
            self.to_k_r = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
            self.to_v_r = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_s = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_g = nn.Linear(head_dim * num_heads + hidden_dim, head_dim * num_heads)
        self.to_out = nn.Linear(head_dim * num_heads, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)             #  Dropout层，少过拟合
        self.ff_mlp = nn.Sequential(                     # 定义一个前馈多层感知机（MLP），它由两个线性层和一个 ReLU 激活函数组成
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        if bipartite:                # 二分图注意力机制
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = nn.LayerNorm(hidden_dim)
        else:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = self.attn_prenorm_x_src
        if has_pos_emb:
            self.attn_prenorm_r = nn.LayerNorm(hidden_dim)
        self.attn_postnorm = nn.LayerNorm(hidden_dim)
        self.ff_prenorm = nn.LayerNorm(hidden_dim)
        self.ff_postnorm = nn.LayerNorm(hidden_dim)
        self.apply(weight_init)         # 权重的初始化，以帮助模型更好地收敛

    def forward(self,
                x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],       # x: 输入特征
                r: Optional[torch.Tensor],                                       # r: 可选的位置嵌入
                edge_index: torch.Tensor) -> torch.Tensor:                       # edge_index: 边索引
        if isinstance(x, torch.Tensor):
            x_src = x_dst = self.attn_prenorm_x_src(x)
        else:
            x_src, x_dst = x
            x_src = self.attn_prenorm_x_src(x_src)
            x_dst = self.attn_prenorm_x_dst(x_dst)
            x = x[1]
        if self.has_pos_emb and r is not None:
            r = self.attn_prenorm_r(r)
        # print(f"Attn.forward, {x.shape=}")
        x = x + self.attn_postnorm(self._attn_block(x_src, x_dst, r, edge_index))
        x = x + self.ff_postnorm(self._ff_block(self.ff_prenorm(x)))
        return x

# 计算每个查询向量与所有键向量之间的相似度，并通过 softmax 函数生成注意力权重，最后使用这些权重来加权求和值向量
    def message(self,
                q_i: torch.Tensor,
                k_j: torch.Tensor,
                v_j: torch.Tensor,
                r: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: Optional[torch.Tensor]) -> torch.Tensor:
        if self.has_pos_emb and r is not None:
            k_j = k_j + self.to_k_r(r).view(-1, self.num_heads, self.head_dim)            # .view(-1, self.num_heads, self.head_dim) 确保维度匹配
            v_j = v_j + self.to_v_r(r).view(-1, self.num_heads, self.head_dim)
        sim = (q_i * k_j).sum(dim=-1) * self.scale  # 缩放因子self.scale用于防止点积结果过大，导致梯度问题
        attn = softmax(sim, index, ptr)
        attn = self.attn_drop(attn)
        return v_j * attn.unsqueeze(-1)

    def update(self,             # 实现一个门控机制，允许模型根据目标节点的特征来调整节点的更新
               inputs: torch.Tensor,
               x_dst: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.num_heads * self.head_dim)
        g = torch.sigmoid(self.to_g(torch.cat([inputs, x_dst], dim=-1)))       # 更新的门控信号
        return inputs + g * (self.to_s(x_dst) - inputs)

#  实现多头自注意力机制
    def _attn_block(self,
                    x_src: torch.Tensor,
                    x_dst: torch.Tensor,
                    r: Optional[torch.Tensor],
                    edge_index: torch.Tensor) -> torch.Tensor:
        q = self.to_q(x_dst).view(-1, self.num_heads, self.head_dim)
        k = self.to_k(x_src).view(-1, self.num_heads, self.head_dim)
        v = self.to_v(x_src).view(-1, self.num_heads, self.head_dim)
        agg = self.propagate(edge_index=edge_index, x_dst=x_dst, q=q, k=k, v=v, r=r)
        return self.to_out(agg)

# 实现一个简单的前馈网络
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff_mlp(x)
