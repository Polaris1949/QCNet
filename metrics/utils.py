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
from typing import Optional, Tuple

import torch
from torch_scatter import gather_csr
from torch_scatter import segment_csr


def topk(
        max_guesses: int,
        pred: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,        # 用于分段压缩稀疏行（CSR）格式的索引
        joint: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:       # 指示是否在处理分段压缩稀疏行（CSR）格式的数据
    max_guesses = min(max_guesses, pred.size(1))
    if max_guesses == pred.size(1):
        if prob is not None:    # 概率归一化
            prob = prob / prob.sum(dim=-1, keepdim=True)
        else:   # 均匀分布的概率分量
            prob = pred.new_ones((pred.size(0), max_guesses)) / max_guesses
        return pred, prob
    else:
        if prob is not None:
            if joint:
                if ptr is None:  # 使用torch.topk直接在归一化后的概率上操作，计算每个样本的平均概率，然后选择前6个最高的概率
                    inds_topk = torch.topk((prob / prob.sum(dim=-1, keepdim=True)).mean(dim=0, keepdim=True),
                                           k=max_guesses, dim=-1, largest=True, sorted=True)[1]
                    inds_topk = inds_topk.repeat(pred.size(0), 1)
                else:    # 使用segment_csr和gather_csr函数处理CSR格式的数据，选择前6个最高的概率。
                    inds_topk = torch.topk(segment_csr(src=prob / prob.sum(dim=-1, keepdim=True), indptr=ptr,
                                                       reduce='mean'),
                                           k=max_guesses, dim=-1, largest=True, sorted=True)[1]
                    inds_topk = gather_csr(src=inds_topk, indptr=ptr)
            else:      # 如果joint参数为False，直接使用torch.topk在归一化后的概率上选择前6个最高的概率
                inds_topk = torch.topk(prob, k=max_guesses, dim=-1, largest=True, sorted=True)[1]
            # 提取相应的预测和概率
            pred_topk = pred[torch.arange(pred.size(0)).unsqueeze(-1).expand(-1, max_guesses), inds_topk]
            prob_topk = prob[torch.arange(pred.size(0)).unsqueeze(-1).expand(-1, max_guesses), inds_topk]
            prob_topk = prob_topk / prob_topk.sum(dim=-1, keepdim=True)         # 对选择的概率进行归一化，使得每个样本的概率之和为1
        else:
            pred_topk = pred[:, :max_guesses]     # 从预测张量pred中选择前6个预测
            prob_topk = pred.new_ones((pred.size(0), max_guesses)) / max_guesses      # 创建一个均匀分布的概率张量，每个预测的概率都是相等的
        return pred_topk, prob_topk


# 用于过滤预测张量pred、目标张量target 以及其他可选张量（如 prob 和 ptr），以便只保留有效的数据
def valid_filter(
        pred: torch.Tensor,
        target: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
        keep_invalid_final_step: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
                                                       torch.Tensor, torch.Tensor]:
    if valid_mask is None: # 创建一个全为 True 的布尔张量，大小与 target 的前几个维度相同
        valid_mask = target.new_ones(target.size()[:-1], dtype=torch.bool)
    if keep_invalid_final_step:
        filter_mask = valid_mask.any(dim=-1)         # 表示每个序列中至少有一个有效步骤
    else:
        filter_mask = valid_mask[:, -1]        # 考虑序列的最后一个步骤
    # 应用过滤掩码
    pred = pred[filter_mask]
    target = target[filter_mask]
    if prob is not None:
        prob = prob[filter_mask]
    valid_mask = valid_mask[filter_mask]
    if ptr is not None:             # 重新计算ptr反映过滤后的序列
        num_nodes_batch = segment_csr(src=filter_mask.long(), indptr=ptr, reduce='sum')
        ptr = num_nodes_batch.new_zeros((num_nodes_batch.size(0) + 1,))
        torch.cumsum(num_nodes_batch, dim=0, out=ptr[1:])
    else:
        ptr = target.new_tensor([0, target.size(0)])   # 创建一个新张量
    return pred, target, prob, valid_mask, ptr
