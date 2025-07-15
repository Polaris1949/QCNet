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
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse

from layers import AttentionLayer
from layers import FourierEmbedding
from layers import MLPLayer
from utils import angle_between_2d_vectors
from utils import bipartite_dense_to_sparse
from utils import weight_init
from utils import wrap_angle



class QCNetDecoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,    # 预测模式数量
                 num_recurrent_steps: int,
                 num_t2m_steps: Optional[int], # 从时间到地图的转换步骤数量
                 pl2m_radius: float,     # 从位置到地图的转换半径
                 a2m_radius: float,      # 从代理到地图的转换半径
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,          # 多头注意力机制中的头数
                 head_dim: int,
                 dropout: float) -> None: # 防止过拟合
        super(QCNetDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_t2m_steps = num_t2m_steps if num_t2m_steps is not None else num_historical_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        input_dim_r_t = 4
        input_dim_r_pl2m = 3
        input_dim_r_a2m = 3

        # 定义解码器的不同组件，包括嵌入层、注意力层和多层感知机层
        # 使用嵌入层将不同的索引和特征映射到高维空间。
        # 使用注意力层处理智能体之间的关系、智能体与地图之间的关系以及地图内部的关系。
        # 使用GRU层处理序列数据并捕捉时间依赖性。
        # 使用MLP层生成预测轨迹的位置和尺度。
        self.mode_emb = nn.Embedding(num_modes, hidden_dim)   # 用于将模式索引映射到嵌入向量
        # 傅里叶嵌入层，用于将不同类型的关系特征映射到高维空间
        self.r_t2m_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2m_emb = FourierEmbedding(input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2m_emb = FourierEmbedding(input_dim=input_dim_r_a2m, hidden_dim=hidden_dim,
                                         num_freq_bands=num_freq_bands)
        # 傅里叶嵌入层，用于将输出特征（可能包括位置和智能体头部信息）映射到高维空间
        self.y_emb = FourierEmbedding(input_dim=output_dim + output_head, hidden_dim=hidden_dim,
                                      num_freq_bands=num_freq_bands)
        # 一个GRU（门控循环单元）层，用于处理序列数据并捕捉时间依赖性
        self.traj_emb = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                               batch_first=False, dropout=0.0, bidirectional=False)
        # GRU层的初始隐藏状态，作为一个可学习的参数
        self.traj_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        # 注意力层的列表，用于在提出阶段处理不同类型的关系
        self.t2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        # 用于在提出阶段处理地图之间的关系的注意力层
        self.m2m_propose_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                     dropout=dropout, bipartite=False, has_pos_emb=False)
        # 这些是注意力层的列表，用于在细化阶段处理不同类型的关系
        self.t2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        # 用于在细化阶段处理地图之间的关系的注意力层
        self.m2m_refine_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                    dropout=dropout, bipartite=False, has_pos_emb=False)
        # 在提出阶段生成位置和尺度的MLP层
        self.to_loc_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim=num_future_steps * output_dim // num_recurrent_steps)
        self.to_scale_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                             output_dim=num_future_steps * output_dim // num_recurrent_steps)
        # 在细化阶段用于生成位置和尺度的MLP层
        self.to_loc_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                          output_dim=num_future_steps * output_dim)
        self.to_scale_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                            output_dim=num_future_steps * output_dim)
        if output_head:
            self.to_loc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,     # 创建一个MLP层，用于在提出阶段预测代理头部的位置。
                                                output_dim=num_future_steps // num_recurrent_steps)
            self.to_conc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,    # 用于在提出阶段预测智能体头部的置信度
                                                 output_dim=num_future_steps // num_recurrent_steps)
            self.to_loc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_future_steps)  # 用于在细化阶段预测智能体头部的位置
            self.to_conc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,     # 用于在细化阶段预测智能体头部的置信度
                                                output_dim=num_future_steps)
        else:
            self.to_loc_propose_head = None
            self.to_conc_propose_head = None
            self.to_loc_refine_head = None
            self.to_conc_refine_head = None
        self.to_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)      # 创建一个MLP层，用于预测一个标量值
        self.apply(weight_init)

    def forward(self,
                data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pos_m = data['agent']['position'][:, self.num_historical_steps - 1, :self.input_dim]  # 从数据中提取智能体的最新（最后一个历史）位置信息。
        head_m = data['agent']['heading'][:, self.num_historical_steps - 1]    # 从数据中提取智能体的最新朝向信息
        head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1)   # 将智能体的朝向角度转换为二维向量表示

        x_t = scene_enc['x_a'].reshape(-1, self.hidden_dim)     # 场景编码中提取与智能体相关的特征x_a，并将其重塑为二维张量，其中每行代表一个代理的特征向量
        x_pl = scene_enc['x_pl'][:, self.num_historical_steps - 1].repeat(self.num_modes, 1)     # 从场景编码中提取与地图多边形相关的特征x_pl，取每个智能体在最后一个历史时间步的特征，沿着第一个维度重复self.num_modes次
        x_a = scene_enc['x_a'][:, -1].repeat(self.num_modes, 1)  # 从场景编码中提取每个智能体在最后一个历史时间步的特征，然后沿着第一个维度重复self.num_modes次
        m = self.mode_emb.weight.repeat(scene_enc['x_a'].size(0), 1)   # 使用模式嵌入层self.mode_emb的权重作为嵌入向量，为每个模式生成一个嵌入表示，沿着第一个维度重复每个嵌入，以匹配输入智能体的数量

        mask_src = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous() # 创建源数据的有效性掩码：从数据中提取智能体的有效性掩码，这个掩码指示了在历史时间步中哪些智能体是有效的
                                                                                           # self.num_historical_steps 指定了需要考虑的历史时间步的数量
        mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False  # 更新源数据的有效性掩码：将源数据掩码中的一部分设置为 False，这通常是为了排除在时间到地图（time-to-map）转换步骤中不会用到的时间步
                                                                              # self.num_t2m_steps 指定了从当前时间点向前看的时间步数量，这些时间步的数据将被保留
        mask_dst = data['agent']['predict_mask'].any(dim=-1, keepdim=True).repeat(1, self.num_modes)  # 创建目标数据的有效性掩码：从数据中提取智能体的预测掩码，这个掩码指示了在预测时间步中哪些智能体是需要进行预测的
                                                                                                      # any(dim=-1)操作用于确定在最后一个维度上是否有任何有效值（即至少有一个未来时间步需要预测）。keepdim=True保持了输出张量的维度不变

        # 分别提取智能体在历史时间步中的位置和朝向，并将其重塑为二维张量
        pos_t = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].reshape(-1, self.input_dim)
        head_t = data['agent']['heading'][:, :self.num_historical_steps].reshape(-1)
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst[:, -1:].unsqueeze(1))  # 使用bipartite_dense_to_sparse函数将源掩码mask_src和目标掩码mask_dst转换为稀疏格式的边索引
        rel_pos_t2m = pos_t[edge_index_t2m[0]] - pos_m[edge_index_t2m[1]]         # 分别计算智能体和地图之间的位置和朝向的相对差异
        rel_head_t2m = wrap_angle(head_t[edge_index_t2m[0]] - head_m[edge_index_t2m[1]])
        r_t2m = torch.stack( # r_t2m是一个堆叠的特征向量，包括相对位置的欧几里得长度、相对位置和代理朝向向量之间的夹角、相对朝向角度，以及一个表示时间步的维度，这些特征被用来捕捉智能体和地图之间的关系
            [torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2m[1]], nbr_vector=rel_pos_t2m[:, :2]),
             rel_head_t2m,
             (edge_index_t2m[0] % self.num_historical_steps) - self.num_historical_steps + 1], dim=-1)
        r_t2m = self.r_t2m_emb(continuous_inputs=r_t2m, categorical_embs=None)   # 将构建的关系特征向量r_t2m传递给关系嵌入层self.r_t2m_emb进行嵌入
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst.unsqueeze(1))   # 再次构建边索引，这次可能包括所有预测模式
        r_t2m = r_t2m.repeat_interleave(repeats=self.num_modes, dim=0)  # 将关系特征向量沿第一个维度重复self.num_modes次，以匹配不同的预测模式

        pos_pl = data['map_polygon']['position'][:, :self.input_dim]  # 提取地图多边形的位置和方向信息
        orient_pl = data['map_polygon']['orientation']
        edge_index_pl2m = radius(            # ：使用radius函数根据位置和半径self.pl2m_radius来构建地图多边形之间的关系
            x=pos_m[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2m_radius,
            batch_x=data['agent']['batch'] if isinstance(data, Batch) else None,         # 区分不同批次的节点
            batch_y=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
            max_num_neighbors=300)
        edge_index_pl2m = edge_index_pl2m[:, mask_dst[edge_index_pl2m[1], 0]]     # 使用目标掩码mask_dst过滤边索引，只保留有效的边
        rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_m[edge_index_pl2m[1]]     # 计算地图多边形之间的相对位置和方向
        rel_orient_pl2m = wrap_angle(orient_pl[edge_index_pl2m[0]] - head_m[edge_index_pl2m[1]])
        r_pl2m = torch.stack(                # r_pl2m是一个堆叠的特征向量，包括相对位置的欧几里得长度、相对位置和代理朝向向量之间的夹角、相对方向角度
            [torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2m[1]], nbr_vector=rel_pos_pl2m[:, :2]),
             rel_orient_pl2m], dim=-1)
        r_pl2m = self.r_pl2m_emb(continuous_inputs=r_pl2m, categorical_embs=None)        # 将构建的关系特征向量r_pl2m传递给关系嵌入层self.r_pl2m_emb进行嵌入
        edge_index_pl2m = torch.cat([edge_index_pl2m + i * edge_index_pl2m.new_tensor(   # 扩展边索引以包含多个预测模式
            [[data['map_polygon']['num_nodes']], [data['agent']['num_nodes']]]) for i in range(self.num_modes)], dim=1)
        r_pl2m = r_pl2m.repeat(self.num_modes, 1)       # 将关系特征向量沿第一个维度重复self.num_modes次，以匹配不同的预测模式

        #  处理智能体之间的关系
        edge_index_a2m = radius_graph(         # 使用radius_graph函数构建代理之间的边，基于智能体的位置和给定的半径self.a2m_radius。loop=False表示不创建自环
            x=pos_m[:, :2],
            r=self.a2m_radius,
            batch=data['agent']['batch'] if isinstance(data, Batch) else None,
            loop=False,
            max_num_neighbors=300)
        edge_index_a2m = edge_index_a2m[:, mask_src[:, -1][edge_index_a2m[0]] & mask_dst[edge_index_a2m[1], 0]] # 使用源掩码mask_src和目标掩码mask_dst来过滤边索引，确保只保留有效的智能体之间的边
        rel_pos_a2m = pos_m[edge_index_a2m[0]] - pos_m[edge_index_a2m[1]]         # 计算智能体之间的相对位置和朝向
        rel_head_a2m = wrap_angle(head_m[edge_index_a2m[0]] - head_m[edge_index_a2m[1]])
        r_a2m = torch.stack(       # 构建智能体之间的关系特征向量
            [torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_a2m[1]], nbr_vector=rel_pos_a2m[:, :2]),
             rel_head_a2m], dim=-1)
        r_a2m = self.r_a2m_emb(continuous_inputs=r_a2m, categorical_embs=None)  # 将构建的关系特征向量r_a2m传递给关系嵌入层self.r_a2m_emb进行嵌入
        edge_index_a2m = torch.cat(               # 扩展边索引以包含多个预测模式
            [edge_index_a2m + i * edge_index_a2m.new_tensor([data['agent']['num_nodes']]) for i in
             range(self.num_modes)], dim=1)
        r_a2m = r_a2m.repeat(self.num_modes, 1)         # 将关系特征向量沿第一个维度重复 self.num_modes 次，以匹配不同的预测模式

        # 构建地图多边形之间的关系
        edge_index_m2m = dense_to_sparse(mask_dst.unsqueeze(2) & mask_dst.unsqueeze(1))[0] # 使用dense_to_sparse函数将目标掩码mask_dst转换为稀疏格式的边索引，用于表示地图多边形之间的关系

        # 是用于存储每个递归步骤预测位置、尺度、头部位置和头部置信度的列表
        locs_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        scales_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        locs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        concs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        for t in range(self.num_recurrent_steps):
            for i in range(self.num_layers):         # m 是模式嵌入的输出，它在每个注意力层中被更新，以反映不同关系的影响
                m = m.reshape(-1, self.hidden_dim)
                m = self.t2m_propose_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)         # 通过三个类型的注意力层（时间到地图t2m，多边形到地图pl2m，智能体到地图a2m）更新模式嵌入
                m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                m = self.pl2m_propose_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                m = self.a2m_propose_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
                m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m = self.m2m_propose_attn_layer(m, None, edge_index_m2m)    # 通过地图间注意力层进一步更新模式嵌入，这里None表示没有关系特征输入
            m = m.reshape(-1, self.num_modes, self.hidden_dim)
            locs_propose_pos[t] = self.to_loc_propose_pos(m)             # 使用 MLP 层生成位置预测
            scales_propose_pos[t] = self.to_scale_propose_pos(m)         # 使用 MLP 层生成尺度预测
            if self.output_head:       # 生成头部位置和置信度预测
                locs_propose_head[t] = self.to_loc_propose_head(m)
                concs_propose_head[t] = self.to_conc_propose_head(m)
        loc_propose_pos = torch.cumsum( # 将之前存储在 locs_propose_pos列表中的所有递归步骤的位置预测张量沿着最后一个维度进行拼接。这将创建一个形状为N是批次大小，M是模式数量，F是未来时间步数量，D是输出维度的张量
            torch.cat(locs_propose_pos, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),  # view函数将拼接后的张量重塑为四维张量，以便于后续处理
            dim=-2)          # 计算沿着未来时间步（dim=-2）的累积和。这将每个时间步的位置预测作为前一个时间步位置和当前预测之和，从而生成一个累积的位置预测序列
        scale_propose_pos = torch.cumsum(
            F.elu_( # F.elu_使用指数线性单元（ELU）激活函数处理尺度预测，alpha=1.0是激活函数的参数
                torch.cat(scales_propose_pos, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
                alpha=1.0) +
            1.0,       # 在激活后的尺度预测上加上1.0，可能是为了确保尺度值的正值
            dim=-2) + 0.1       # 最后在累积的尺度预测上加上一个常数0.1，这可能是为了提供一个最小的尺度值，确保尺度不会太小
        if self.output_head:
            loc_propose_head = torch.cumsum(torch.tanh(torch.cat(locs_propose_head, dim=-1).unsqueeze(-1)) * math.pi,  # torch.tanh(torch.cat(locs_propose_head, dim=-1).unsqueeze(-1)) * math.pi：
                                            dim=-2)  # 首先将头部位置预测拼接并增加一个维度，然后通过双曲正切（tanh）函数进行激活，使其范围在-1到1之间，最后乘以π将其转换为角度表示
            conc_propose_head = 1.0 / (torch.cumsum(F.elu_(torch.cat(concs_propose_head, dim=-1).unsqueeze(-1)) + 1.0, # # 将头部置信度预测拼接并增加一个维度，通过指数线性单元激活函数处理，并加上 1.0确保尺度值的正值
                                                    dim=-2) + 0.02)       # 通过对累积和取倒数并加上一个小常数（0.02），生成头部置信度预测
            m = self.y_emb(torch.cat([loc_propose_pos.detach(),    # 将位置和头部位置的拼接结果传递给输出嵌入层self.y_emb进行嵌入
                                      wrap_angle(loc_propose_head.detach())], dim=-1).view(-1, self.output_dim + 1))
        else:
            loc_propose_head = loc_propose_pos.new_zeros((loc_propose_pos.size(0), self.num_modes,  # 初始化头部预测：初始化为与位置预测和尺度预测相同大小的零张量
                                                          self.num_future_steps, 1))
            conc_propose_head = scale_propose_pos.new_zeros((scale_propose_pos.size(0), self.num_modes,
                                                             self.num_future_steps, 1))
            m = self.y_emb(loc_propose_pos.detach().view(-1, self.output_dim))        # 只将位置预测传递给输出嵌入层self.y_emb进行嵌入
        m = m.reshape(-1, self.num_future_steps, self.hidden_dim).transpose(0, 1)     # 将嵌入后的特征m重塑并转置，以适应GRU层的输入格式
        m = self.traj_emb(m, self.traj_emb_h0.unsqueeze(1).repeat(1, m.size(1), 1))[1].squeeze(0) # 将特征 m 通过GRU层进行处理，[1].squeeze(0)获取GRU的最后一个输出并去除多余的维度
        for i in range(self.num_layers):   # 通过三个类型的注意力层（时间到地图t2m，多边形到地图pl2m，代理到地图a2m）迭代地更新特征m
            m = self.t2m_refine_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)    # 在每个注意力层中，特征m被重塑和转置，以匹配注意力层的输入要求
            m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m = self.pl2m_refine_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
            m = self.a2m_refine_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
            m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        m = self.m2m_refine_attn_layer(m, None, edge_index_m2m)     # 通过地图间注意力层进一步更新特征m
        m = m.reshape(-1, self.num_modes, self.hidden_dim)
        loc_refine_pos = self.to_loc_refine_pos(m).view(-1, self.num_modes, self.num_future_steps, self.output_dim)  # 使用MLP层生成细化的位置预测
        loc_refine_pos = loc_refine_pos + loc_propose_pos.detach()      # 将细化的位置预测与提出的预测相加，得到最终的位置预测
        scale_refine_pos = F.elu_(             # 使用MLP层生成细化的尺度预测，并通过ELU激活函数和常数偏移来处理
            self.to_scale_refine_pos(m).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
            alpha=1.0) + 1.0 + 0.1
        if self.output_head:
            loc_refine_head = torch.tanh(self.to_loc_refine_head(m).unsqueeze(-1)) * math.pi         # 使用MLP层生成细化的头部位置预测，并增加一个维度，通过双曲正切函数激活，使其范围在-1到1之间，然后乘以π将其转换为角度表示
            loc_refine_head = loc_refine_head + loc_propose_head.detach()    # 将细化的头部位置预测与提出的预测相加，得到最终的头部位置预测
            conc_refine_head = 1.0 / (F.elu_(self.to_conc_refine_head(m).unsqueeze(-1)) + 1.0 + 0.02)  # 通过指数线性单元激活函数处理，并加上常数偏移，通过对激活后的置信度预测取倒数，生成最终的头部置信度预测
        else:
            loc_refine_head = loc_refine_pos.new_zeros((loc_refine_pos.size(0), self.num_modes, self.num_future_steps,
                                                        1))  # 初始化头部预测：初始化为与位置预测和尺度预测相同大小的零张量
            conc_refine_head = scale_refine_pos.new_zeros((scale_refine_pos.size(0), self.num_modes,
                                                           self.num_future_steps, 1))
        pi = self.to_pi(m).squeeze(-1) # 使用MLP层 elf.to_pi生成一个标量预测pi，然后去除最后一个维度

        return {
            'loc_propose_pos': loc_propose_pos,
            'scale_propose_pos': scale_propose_pos,
            'loc_propose_head': loc_propose_head,
            'conc_propose_head': conc_propose_head,
            'loc_refine_pos': loc_refine_pos,
            'scale_refine_pos': scale_refine_pos,
            'loc_refine_head': loc_refine_head,
            'conc_refine_head': conc_refine_head,
            'pi': pi,
        }
