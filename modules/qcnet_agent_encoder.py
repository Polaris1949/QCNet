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
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import subgraph

from layers.attention_layer import AttentionLayer
from layers.fourier_embedding import FourierEmbedding
from utils import angle_between_2d_vectors
from utils import weight_init
from utils import wrap_angle


class QCNetAgentEncoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 time_span: Optional[int], # 时间跨度
                 pl2a_radius: float,       # 从位置到注意力层的半径
                 a2a_radius: float,        # 注意力到注意力层的半径
                 num_freq_bands: int,      # 频带的数量
                 num_layers: int,
                 num_heads: int,           # 多头注意力机制中的头数
                 head_dim: int,            # 每个注意力头的维度
                 dropout: float) -> None:  # 防止过拟合
        super(QCNetAgentEncoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        if dataset == 'argoverse_v2':
            input_dim_x_a = 4
            input_dim_r_t = 4
            input_dim_r_pl2a = 3
            input_dim_r_a2a = 3
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))

        if dataset == 'argoverse_v2':
            self.type_a_emb = nn.Embedding(10, hidden_dim)          # 嵌入层将用于将类型索引映射到固定维度的向量表示
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))
        # 捕捉输入数据的周期性和频率特性
        self.x_a_emb = FourierEmbedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_t_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2a_emb = FourierEmbedding(input_dim=input_dim_r_pl2a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r_a2a, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.t_attn_layers = nn.ModuleList(                     # bipartite表示注意力机制是二分图注意力
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2a_attn_layers = nn.ModuleList(                  # has_pos_emb表示注意力层将包含位置嵌入
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.apply(weight_init)    # 初始化权重

    def forward(self,
                data: HeteroData,      # 异构数据
                map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mask = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()    # 从数据中提取智能体的有效性掩码，并确保它只包含历史步骤的数量。contiguous() 方法用于确保张量在内存中是连续的
        pos_a = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].contiguous()        # 提取智能体的位置信息
        motion_vector_a = torch.cat([pos_a.new_zeros(data['agent']['num_nodes'], 1, self.input_dim),
                                     pos_a[:, 1:] - pos_a[:, :-1]], dim=1)     # 计算智能体的运动向量
        head_a = data['agent']['heading'][:, :self.num_historical_steps].contiguous()            # 提取智能体的朝向信息
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)           # 将智能体的朝向转换为向量表示
        pos_pl = data['map_polygon']['position'][:, :self.input_dim].contiguous()     # 提取地图多边形的位置信息
        orient_pl = data['map_polygon']['orientation'].contiguous()               # 提取地图多边形的朝向信息
        if self.dataset == 'argoverse_v2':    # 处理输入数据，提取和转换智能体的速度和类型信息，并将类型信息转换为嵌入向量
            vel = data['agent']['velocity'][:, :self.num_historical_steps, :self.input_dim].contiguous()  # 从数据中提取智能体的速度信息
            length = width = height = None  # 尺寸设置
            categorical_embs = [
                self.type_a_emb(data['agent']['type'].long()).repeat_interleave(repeats=self.num_historical_steps, # 用self.type_a_emb嵌入层将代理的类型转换为嵌入向量。
                                                                                dim=0),  #  然后使用 repeat_interleave 方法在第0维上重复每个嵌入向量self.num_historical_steps次，以匹配历史步骤的数量
            ]
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        if self.dataset == 'argoverse_v2':
            x_a = torch.stack(               # 多个计算得到的特征向量堆叠形成一个新的特征向量
                [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),     # 表示智能体在两个连续时间步之间的移动距离
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]),   # 计算智能体的运动向量和代理朝向向量之间的夹角
                 torch.norm(vel[:, :, :2], p=2, dim=-1),        # 表示智能体在两个连续时间步之间的平均速度
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel[:, :, :2])], dim=-1) # 计算智能体速度向量和代理朝向向量之间的夹角
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        # view()方法用于重新塑形张量的形状而不改变其数据
        x_a = self.x_a_emb(continuous_inputs=x_a.view(-1, x_a.size(-1)), categorical_embs=categorical_embs) # 将计算得到的特征向量x_a通过Fourier嵌入层进行嵌入
        x_a = x_a.view(-1, self.num_historical_steps, self.hidden_dim)       # 嵌入后的特征向量重新塑形，以匹配历史步骤的数量和隐藏层的维度

        pos_t = pos_a.reshape(-1, self.input_dim)      # 将智能体的位置张量pos_a重塑为每个智能体一个位置向量的格式。
        head_t = head_a.reshape(-1)              #  将智能体的朝向角度张量head_a重塑为一维张量
        head_vector_t = head_vector_a.reshape(-1, 2)   # 将智能体的朝向向量张量head_vector_a重塑为每个智能体一个二维向量的格式
        mask_t = mask.unsqueeze(2) & mask.unsqueeze(1) # 将有效性掩码mask在两个维度上扩展并进行逐元素的逻辑与操作，以得到一个表示智能体之间关系的掩码
        edge_index_t = dense_to_sparse(mask_t)[0]      # 将密集的掩码矩阵转换为稀疏的边索引格式
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]    # 过滤边索引，只保留源节点索引小于目标节点索引的边
        edge_index_t = edge_index_t[:, edge_index_t[1] - edge_index_t[0] <= self.time_span]       # 进一步过滤边索引，只保留时间跨度在模型考虑范围内的边
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]    # 计算相对位置
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])    # 计算相对朝向，并使用wrap_angle函数进行角度的包装
        r_t = torch.stack(                # 将计算得到的相对位置的欧几里得长度、相对位置和朝向向量之间的夹角、相对朝向角度以及节点索引差堆叠成一个新的特征向量
            [torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
             rel_head_t,
             edge_index_t[0] - edge_index_t[1]], dim=-1)
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None) # 将构建的关系特征向量r_t传递给关系时间嵌入层self.r_t_emb进行嵌入

        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)   # 首先将智能体的位置张量pos_a进行转置，使得历史步骤成为第一个维度，然后重塑为一维索引，每个代理一个位置向量
        head_s = head_a.transpose(0, 1).reshape(-1)           # 将智能体的朝向角度张量head_a进行转置并重塑，使得历史步骤成为第一个维度
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)   # 将智能体的朝向向量张量head_vector_a进行转置并重塑，使得历史步骤成为第一个维度，每个代理一个二维朝向向量
        mask_s = mask.transpose(0, 1).reshape(-1)        # 将有效掩码进行转置并重塑，使得历史步骤成为第一个维度
        pos_pl = pos_pl.repeat(self.num_historical_steps, 1)     # 将地图多边形的位置张量pos_pl在第一个维度上重复self.num_historical_steps次，以匹配代理的历史步骤数量
        orient_pl = orient_pl.repeat(self.num_historical_steps)
        if isinstance(data, Batch):
            batch_s = torch.cat([data['agent']['batch'] + data.num_graphs * t            # 使用列表推导式和torch.cat来连接多个批次索引张量，确保不同时间步的智能体有不同的批次索引
                                 for t in range(self.num_historical_steps)], dim=0)
            batch_pl = torch.cat([data['map_polygon']['batch'] + data.num_graphs * t
                                  for t in range(self.num_historical_steps)], dim=0)
        else:
            batch_s = torch.arange(self.num_historical_steps,         # 使用torch.arange生成一个从0到self.num_historical_steps - 1的序列
                                   device=pos_a.device).repeat_interleave(data['agent']['num_nodes'])
            batch_pl = torch.arange(self.num_historical_steps,
                                    device=pos_pl.device).repeat_interleave(data['map_polygon']['num_nodes'])
        edge_index_pl2a = radius(x=pos_s[:, :2], y=pos_pl[:, :2], r=self.pl2a_radius, batch_x=batch_s, batch_y=batch_pl,
                                 max_num_neighbors=300)   # 使用radius函数根据位置和半径self.pl2a_radius来构建智能体和地图多边形之间的边。max_num_neighbors=300限制了每个节点的最大邻居数量。
        edge_index_pl2a = edge_index_pl2a[:, mask_s[edge_index_pl2a[1]]]  # 使用掩码mask_s过滤边，只保留有效的边
# 计算智能体和地图多边形之间的相对位置和方向
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])
        r_pl2a = torch.stack(        # 将相对位置的欧几里得长度、相对位置和朝向向量之间的夹角、相对方向角度堆叠成一个新的特征向量
            [torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]),
             rel_orient_pl2a], dim=-1)
        r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a, categorical_embs=None) # 将构建的关系特征向量r_pl2a传递给智能体和地图多边形之间的关系嵌入层self.r_pl2a_emb进行嵌入。
        edge_index_a2a = radius_graph(x=pos_s[:, :2], r=self.a2a_radius, batch=batch_s, loop=False, # 使用radius函数根据位置和半径self.pl2a_radius来构建智能体和智能体之间的边
                                      max_num_neighbors=300)                                        # loop=False 表示不添加自循环
        edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0]         # 使用subgraph函数和掩码mask_s过滤边，只保留有效的边
        # 计算智能体之间的相对位置和方向
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(         # 将相对位置的欧几里得长度、相对位置和朝向向量之间的夹角、相对方向角度堆叠成一个新的特征向量
            [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
             rel_head_a2a], dim=-1)
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)     # 将构建的关系特征向量r_a2a传递给智能体之间的关系嵌入层self.r_a2a_emb进行嵌入

# 通过多层注意力机制来更新智能体的特征表示 x_a
        for i in range(self.num_layers):
            x_a = x_a.reshape(-1, self.hidden_dim)    # 将智能体的特征表示x_a重塑为二维张量
            x_a = self.t_attn_layers[i](x_a, r_t, edge_index_t)          # 将重塑后的x_a、关系特征r_t和边索引edge_index_t传递给第i层时间注意力层进行处理
            x_a = x_a.reshape(-1, self.num_historical_steps,             # 处理后的 x_a 再次重塑，以恢复其原始的时间步维度
                              self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            x_a = self.pl2a_attn_layers[i]((map_enc['x_pl'].transpose(0, 1).reshape(-1, self.hidden_dim), x_a), r_pl2a,
                                           edge_index_pl2a)              # 将地图多边形的特征map_enc['x_pl']和智能体的特征x_a传递给第i层智能体与地图多边形注意力层，同时传递关系特征r_pl2a和边索引edge_index_pl2a
            x_a = self.a2a_attn_layers[i](x_a, r_a2a, edge_index_a2a)    # 将智能体的特征x_a传递给第i层智能体间注意力层，同时传递关系特征r_a2a和边索引edge_index_a2a
            x_a = x_a.reshape(self.num_historical_steps, -1, self.hidden_dim).transpose(0, 1)   # 在所有注意力层处理完成后，将x_a最终重塑回其原始的时间步维度

        return {'x_a': x_a}
