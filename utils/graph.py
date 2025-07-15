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
from typing import List, Optional, Tuple, Union

import torch
from torch_geometric.utils import coalesce
from torch_geometric.utils import degree



def add_edges(
        from_edge_index: torch.Tensor,     # 表示要添加的边的起始节点索引
        to_edge_index: torch.Tensor,       # 表示当前图中的边的终点节点索引
        from_edge_attr: Optional[torch.Tensor] = None,     # 表示要添加的边的属性
        to_edge_attr: Optional[torch.Tensor] = None,       # 表示当前图中边的属性
        replace: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:       # 指示在添加边时是否替换已存在的边
    from_edge_index = from_edge_index.to(device=to_edge_index.device, dtype=to_edge_index.dtype)   # 确保数据类型一致
    mask = ((to_edge_index[0].unsqueeze(-1) == from_edge_index[0].unsqueeze(0)) &             # 创建一个掩码 mask，用于标识from_edge_index中的边是否已经在to_edge_index中存在
            (to_edge_index[1].unsqueeze(-1) == from_edge_index[1].unsqueeze(0)))
    if replace:
        to_mask = mask.any(dim=1)         # 指定维度计算掩码，以确定需要被替代的边
        if from_edge_attr is not None and to_edge_attr is not None:      # 将新边属性替换旧边属性
            from_edge_attr = from_edge_attr.to(device=to_edge_attr.device, dtype=to_edge_attr.dtype)
            to_edge_attr = torch.cat([to_edge_attr[~to_mask], from_edge_attr], dim=0)
        to_edge_index = torch.cat([to_edge_index[:, ~to_mask], from_edge_index], dim=1)  #新to_edge_index以包含新添加的边
    else: #将新边添加到图中，不替换任何已存在的边
        from_mask = mask.any(dim=0)
        if from_edge_attr is not None and to_edge_attr is not None:
            from_edge_attr = from_edge_attr.to(device=to_edge_attr.device, dtype=to_edge_attr.dtype)
            to_edge_attr = torch.cat([to_edge_attr, from_edge_attr[~from_mask]], dim=0) # 将新边的属性添加到to_edge_attr中
        to_edge_index = torch.cat([to_edge_index, from_edge_index[:, ~from_mask]], dim=1)
    return to_edge_index, to_edge_attr


# 合并多个边索引和边属性列表
def merge_edges(
        edge_indices: List[torch.Tensor],    # 一个包含多个边索引张量的列表
        edge_attrs: Optional[List[torch.Tensor]] = None,         # 一个可选的包含多个边属性张量的列表
        reduce: str = 'add') -> Tuple[torch.Tensor, Optional[torch.Tensor]]:  # 一个字符串，指定在合并过程中如何处理重复的边
    # torch.cat用于将多个张量沿着已存在的维度进行连接
    edge_index = torch.cat(edge_indices, dim=1)       # 将所有边索引张量沿着第一个维度连接起来，形成一个新的边索引张量
    if edge_attrs is not None:
        edge_attr = torch.cat(edge_attrs, dim=0)      # 将所有边属性张量沿着第一个维度连接起来，形成一个新的边属性张量
    else:
        edge_attr = None
    # coalesce 函数通常用于确保边索引中没有重复的边，并且对边属性进行适当的约简操作
    return coalesce(edge_index=edge_index, edge_attr=edge_attr, reduce=reduce) # 合并边索引和边属性，并根据reduce参数指定的方式处理重复的边。


# 创建一个完全图的边索引
def complete_graph(
        num_nodes: Union[int, Tuple[int, int]],     # 表示图中节点的数量
        ptr: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,       # 指定二分图中每个部分的起始节点索引
        loop: bool = False,                         # 示是否在图中包含自环
        device: Optional[Union[torch.device, str]] = None) -> torch.Tensor:     # 指定张量的存储位置
    if ptr is None:    # 使用torch.cartesian_prod生成所有可能的节点对，形成完全图的边索引
        if isinstance(num_nodes, int):     # 检查 num_nodes 是否是一个整数
            num_src, num_dst = num_nodes, num_nodes
        else:  # 果 num_nodes 不是一个整数，那么它会被认为是一个包含两个元素的元组或列表
            num_src, num_dst = num_nodes
        edge_index = torch.cartesian_prod(torch.arange(num_src, dtype=torch.long, device=device), # 生成了所有可能的源节点和目标节点的组合
                                          torch.arange(num_dst, dtype=torch.long, device=device)).t()
    else:     # 则使用它来构建二分图的边索引，每个部分的节点对都加上相应的偏移量
        if isinstance(ptr, torch.Tensor):
            ptr_src, ptr_dst = ptr, ptr   # ptr_src 和 ptr_dst 都设置为ptr，源节点和目标节点的偏移量数组是相同的
            num_src_batch = num_dst_batch = ptr[1:] - ptr[:-1]
        else:
            ptr_src, ptr_dst = ptr
            num_src_batch = ptr_src[1:] - ptr_src[:-1]    # 计算每个批次中的源节点和目标节点的数量
            num_dst_batch = ptr_dst[1:] - ptr_dst[:-1]
        edge_index = torch.cat(
            [torch.cartesian_prod(torch.arange(num_src, dtype=torch.long, device=device),  # 生成每个批次内部所有可能的节点对，通过加上偏移量 p 来构建全局的边索引
                                  torch.arange(num_dst, dtype=torch.long, device=device)) + p      # p 是由 torch.stack([ptr_src, ptr_dst], dim=1) 生成的，它将源节点和目标节点的偏移量堆叠起来
             for num_src, num_dst, p in zip(num_src_batch, num_dst_batch, torch.stack([ptr_src, ptr_dst], dim=1))],        # 在列表推导式中遍历每个批次的源节点数、目标节点数和偏移量
            dim=0)
        edge_index = edge_index.t() # 生成的边索引张量转置
    if isinstance(num_nodes, int) and not loop: # 如果 num_nodes是一个整数并且loop参数为False，则移除自环（即源节点和目标节点相同的边）
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    return edge_index.contiguous()


# 接受一个表示二分图邻接矩阵的PyTorch张量adj并将其转换为稀疏表示形式
def bipartite_dense_to_sparse(adj: torch.Tensor) -> torch.Tensor:
    index = adj.nonzero(as_tuple=True)         # 使用 nonzero 函数找出adj张量中所有非零元素的索引。as_tuple=True参数使得返回的索引以元组的形式呈现，而不是一个张量
    if len(index) == 3:
        batch_src = index[0] * adj.size(1)      # 计算源节点和目标节点的全局索引
        batch_dst = index[0] * adj.size(2)
        index = (batch_src + index[1], batch_dst + index[2])    # 将局部索引转换为全局索引
    return torch.stack(index, dim=0)  # 使用torch.stack函数将两个索引数组堆叠起来，形成一个二维张量，每一列代表一条边的源节点和目标节点索引

#  用于将一个批次的张量拆分成单独的张量，每个张量对应批次中的一个元素
def unbatch(
        src: torch.Tensor, # 一个批次的张量，其中包含了多个元素
        batch: torch.Tensor, # 一个一维张量，包含了每个元素所属批次的索引
        dim: int = 0) -> List[torch.Tensor]:    # 指定 src 张量中要拆分的维度，默认为0
    sizes = degree(batch, dtype=torch.long).tolist()   # 计算每个批次元素的大小
    return src.split(sizes, dim)  # 使用 split 方法将 src 张量沿着指定的维度 dim 拆分成多个张量
