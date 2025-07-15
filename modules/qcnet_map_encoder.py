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
from typing import Dict

import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from layers.attention_layer import AttentionLayer
from layers.fourier_embedding import FourierEmbedding
from utils import angle_between_2d_vectors
from utils import merge_edges
from utils import weight_init
from utils import wrap_angle



class QCNetMapEncoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(QCNetMapEncoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.pl2pl_radius = pl2pl_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        if dataset == 'argoverse_v2':
            if input_dim == 2:       # 表明每个地图多边形只有两个特征
                input_dim_x_pt = 1 # 表明每个多边形只有两个特征
                input_dim_x_pl = 0 # 多边形的特征维度为0
                input_dim_r_pt2pl = 3 # 点到多边形关系的维度为3
                input_dim_r_pl2pl = 3 # 多边形到多边形关系的维度为3
            elif input_dim == 3:
                input_dim_x_pt = 2
                input_dim_x_pl = 1
                input_dim_r_pt2pl = 4
                input_dim_r_pl2pl = 4
            else:
                raise ValueError('{} is not a valid dimension'.format(input_dim))
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))

        if dataset == 'argoverse_v2':       # 初始化嵌入层
            self.type_pt_emb = nn.Embedding(17, hidden_dim)
            self.side_pt_emb = nn.Embedding(3, hidden_dim)
            self.type_pl_emb = nn.Embedding(4, hidden_dim)
            self.int_pl_emb = nn.Embedding(3, hidden_dim)
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))
        self.type_pl2pl_emb = nn.Embedding(5, hidden_dim)     # 将多边形之间的交互类型映射到高维空间中的向量表示
        # 用于嵌入点的特征。傅里叶嵌入能够捕捉输入数据的周期性和频率特性
        self.x_pt_emb = FourierEmbedding(input_dim=input_dim_x_pt, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        # 用于嵌入多边形的特征
        self.x_pl_emb = FourierEmbedding(input_dim=input_dim_x_pl, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pt2pl_emb = FourierEmbedding(input_dim=input_dim_r_pt2pl, hidden_dim=hidden_dim, # 用于嵌入点与多边形之间的关系特征
                                            num_freq_bands=num_freq_bands)
        self.r_pl2pl_emb = FourierEmbedding(input_dim=input_dim_r_pl2pl, hidden_dim=hidden_dim, # 用于嵌入多边形与多边形之间的关系特征
                                            num_freq_bands=num_freq_bands)
        self.pt2pl_layers = nn.ModuleList(        # 包含多个注意力层的列表，用于处理点到多边形的关系
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2pl_layers = nn.ModuleList(        # 包含多个注意力层的列表，用于处理多边形之间的关系
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.apply(weight_init)  # 权重初始化

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        pos_pt = data['map_point']['position'][:, :self.input_dim].contiguous() # 从数据中提取地图点的位置信息，并且下确保只包含所需的维度
        orient_pt = data['map_point']['orientation'].contiguous()         # 从数据中提取地图点的方向信息，并确保张量在内存中是连续的
        pos_pl = data['map_polygon']['position'][:, :self.input_dim].contiguous()       # 从数据中提取地图多边形的位置信息
        orient_pl = data['map_polygon']['orientation'].contiguous()           # 从数据中提取地图多边形的方向信息
        orient_vector_pl = torch.stack([orient_pl.cos(), orient_pl.sin()], dim=-1)     # 将多边形的方向角度转换为二维向量表示

        if self.dataset == 'argoverse_v2':
            if self.input_dim == 2:          # 表示每个地图多边形的初始特征数量
                x_pt = data['map_point']['magnitude'].unsqueeze(-1)    # 提取地图点的magnitude特征，并将其扩展为二维张量
                x_pl = None
            elif self.input_dim == 3:     # 提取地图点的magnitude和height特征，并将它们堆叠为二维张量
                x_pt = torch.stack([data['map_point']['magnitude'], data['map_point']['height']], dim=-1)
                x_pl = data['map_polygon']['height'].unsqueeze(-1)
            else:
                raise ValueError('{} is not a valid dimension'.format(self.input_dim))
            x_pt_categorical_embs = [self.type_pt_emb(data['map_point']['type'].long()),    # 为地图点的类型和侧面创建类别嵌入。这些嵌入将离散的类别特征映射到连续的向量空间中
                                     self.side_pt_emb(data['map_point']['side'].long())]
            x_pl_categorical_embs = [self.type_pl_emb(data['map_polygon']['type'].long()),        # 为地图多边形的类型和是否为交点创建类别嵌入
                                     self.int_pl_emb(data['map_polygon']['is_intersection'].long())]
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        x_pt = self.x_pt_emb(continuous_inputs=x_pt, categorical_embs=x_pt_categorical_embs)   # 使用傅里叶嵌入层处理地图点的连续和类别嵌入特征
        x_pl = self.x_pl_emb(continuous_inputs=x_pl, categorical_embs=x_pl_categorical_embs)   # 使用傅里叶嵌入层处理地图多边形的连续和类别嵌入特征

        edge_index_pt2pl = data['map_point', 'to', 'map_polygon']['edge_index']      # 从数据中提取点到多边形的边索引，这些索引用于确定哪些点与哪些多边形有关系
        rel_pos_pt2pl = pos_pt[edge_index_pt2pl[0]] - pos_pl[edge_index_pt2pl[1]]      # 计算点和多边形之间的相对位置，通过从点的位置中减去相应多边形的位置来实现的
        rel_orient_pt2pl = wrap_angle(orient_pt[edge_index_pt2pl[0]] - orient_pl[edge_index_pt2pl[1]])   # 计算点和多边形之间的相对方向，使用 wrap_angle 函数来处理角度差异
        if self.input_dim == 2:
            r_pt2pl = torch.stack( # 关系特征向量
                [torch.norm(rel_pos_pt2pl[:, :2], p=2, dim=-1),       # 点和多边形之间的欧几里得距离
                 angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pt2pl[1]],  # 点的位置向量和多边形的方向向量之间的夹角
                                          nbr_vector=rel_pos_pt2pl[:, :2]),
                 rel_orient_pt2pl], dim=-1)             # 点和多边形之间的相对方向
        elif self.input_dim == 3:
            r_pt2pl = torch.stack(
                [torch.norm(rel_pos_pt2pl[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pt2pl[1]],
                                          nbr_vector=rel_pos_pt2pl[:, :2]),
                 rel_pos_pt2pl[:, -1],
                 rel_orient_pt2pl], dim=-1)
        else:
            raise ValueError('{} is not a valid dimension'.format(self.input_dim))
        r_pt2pl = self.r_pt2pl_emb(continuous_inputs=r_pt2pl, categorical_embs=None)    # 将构建的关系特征向量传递给点到多边形的关系嵌入层self.r_pt2pl_emb进行嵌入

        edge_index_pl2pl = data['map_polygon', 'to', 'map_polygon']['edge_index']      # 提取多边形到多边形的边索引
        edge_index_pl2pl_radius = radius_graph(x=pos_pl[:, :2], r=self.pl2pl_radius,   # 使用radius_graph函数根据多边形的位置和给定的半径self.pl2pl_radius来构建多边形之间的关系
                                               batch=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
                                               loop=False, max_num_neighbors=300)      # loop=False表示不创建自环，max_num_neighbors=300限制了每个多边形的最大邻居数量
        type_pl2pl = data['map_polygon', 'to', 'map_polygon']['type']      # 提取多边形到多边形之间的关系类型
        type_pl2pl_radius = type_pl2pl.new_zeros(edge_index_pl2pl_radius.size(1), dtype=torch.uint8)     # 创建一个与edge_index_pl2pl_radius相同大小的零张量，用于存储关系类型的半径信息
        edge_index_pl2pl, type_pl2pl = merge_edges(edge_indices=[edge_index_pl2pl_radius, edge_index_pl2pl],  # 使用merge_edges函数合并基于半径和类型的边索引
                                                   edge_attrs=[type_pl2pl_radius, type_pl2pl], reduce='max')
        # 计算多边形之间的相对位置和方向
        rel_pos_pl2pl = pos_pl[edge_index_pl2pl[0]] - pos_pl[edge_index_pl2pl[1]]
        rel_orient_pl2pl = wrap_angle(orient_pl[edge_index_pl2pl[0]] - orient_pl[edge_index_pl2pl[1]])
        if self.input_dim == 2:
            r_pl2pl = torch.stack(       # 构建关系特征向量
                [torch.norm(rel_pos_pl2pl[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pl2pl[1]],   # 多边形的位置向量和另一个多边形的方向向量之间的夹角
                                          nbr_vector=rel_pos_pl2pl[:, :2]),
                 rel_orient_pl2pl], dim=-1)
        elif self.input_dim == 3:
            r_pl2pl = torch.stack(
                [torch.norm(rel_pos_pl2pl[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pl2pl[1]],
                                          nbr_vector=rel_pos_pl2pl[:, :2]),
                 rel_pos_pl2pl[:, -1],
                 rel_orient_pl2pl], dim=-1)
        else:
            raise ValueError('{} is not a valid dimension'.format(self.input_dim))
        # 将构建的关系特征向量传递给多边形到多边形的关系嵌入层self.r_pl2pl_emb进行嵌入。同时，传递类别嵌入type_pl2pl作为关系特征的一部分
        r_pl2pl = self.r_pl2pl_emb(continuous_inputs=r_pl2pl, categorical_embs=[self.type_pl2pl_emb(type_pl2pl.long())])

        for i in range(self.num_layers):
            x_pl = self.pt2pl_layers[i]((x_pt, x_pl), r_pt2pl, edge_index_pt2pl)     # 在点到多边形的注意力层中更新多边形的特征表示：使用点的特征、多边形的特征、关系特征和边索引
            x_pl = self.pl2pl_layers[i](x_pl, r_pl2pl, edge_index_pl2pl)      # 在多边形到多边形的注意力层中进一步更新多边形的特征表示
        x_pl = x_pl.repeat_interleave(repeats=self.num_historical_steps,      # 将多边形的特征沿第一个维度重复self.num_historical_steps次，以匹配历史时间步的数量
                                      dim=0).reshape(-1, self.num_historical_steps, self.hidden_dim) # 将重复的特征重塑为三维张量，其中包含批次大小、历史时间步数量和隐藏层维度

        return {'x_pt': x_pt, 'x_pl': x_pl}
