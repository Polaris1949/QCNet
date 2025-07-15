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

import torch


# 计算两个二维向量之间的角度
def angle_between_2d_vectors(
        ctr_vector: torch.Tensor,        # 表示中心向量的张量
        nbr_vector: torch.Tensor) -> torch.Tensor:         # 表示邻接向量的张量
    # torch.atan2用于计算两个数值之间的角度。它考虑了叉积的符号，因此可以返回正确的角度，即使点积为零
    return torch.atan2(ctr_vector[..., 0] * nbr_vector[..., 1] - ctr_vector[..., 1] * nbr_vector[..., 0],   # 计算两个向量的叉积，这将用于确定角度的正负
                       (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1))   # 计算两个向量的点积，这将用于确定角度的大小


def angle_between_3d_vectors(
        ctr_vector: torch.Tensor,
        nbr_vector: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.cross(ctr_vector, nbr_vector, dim=-1).norm(p=2, dim=-1),       # 使用torch.cross计算两个向量的叉积。它垂直于原始的两个向量，并且其长度等于两个向量构成的平行四边形的面积，.norm计算叉积的模，
                       (ctr_vector * nbr_vector).sum(dim=-1))  # 计算两个向量的点积


def side_to_directed_lineseg(
        query_point: torch.Tensor,  # 查询点的坐标
        start_point: torch.Tensor,  # 有向线段的起点坐标
        end_point: torch.Tensor) -> str:      # 有向线段的终点坐标
    cond = ((end_point[0] - start_point[0]) * (query_point[1] - start_point[1]) -    # 使用向量叉积的公式计算一个条件值cond
            (end_point[1] - start_point[1]) * (query_point[0] - start_point[0]))     # 叉积的符号可以告诉点是在向量从起点到终点的顺时针方向还是逆时针方向
    if cond > 0:
        return 'LEFT'
    elif cond < 0:
        return 'RIGHT'
    else:
        return 'CENTER'


def wrap_angle(
        angle: torch.Tensor,
        min_val: float = -math.pi, # 包装范围的最小值
        max_val: float = math.pi) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)   # angle+max_val：输入的角度向正方向偏移的值，使用%运算符计算偏移角度在0到2π* (max_val - min_val)范围内的模
    # min_val + ...: 将模值调整回 [min_val, max_val] 范围