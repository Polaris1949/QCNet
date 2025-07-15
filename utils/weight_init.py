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
import torch.nn as nn


def weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):     # 检查模型中的层m是否是一个全连接层
        nn.init.xavier_uniform_(m.weight)   # 使用Xavier均匀初始化方法初始化权重
        if m.bias is not None: # 有偏置项，将其初始化为零
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):      # 检查模型中的层 m 是否是卷积层（一维、二维或三维）的一个实例
        fan_in = m.in_channels / m.groups               # 计算卷积层的“fan in”和“fan out”
        fan_out = m.out_channels / m.groups             # 值用于计算权重的初始化范围
        bound = (6.0 / (fan_in + fan_out)) ** 0.5       # 计算权重初始化的界限，这是基于He初始化（也称为Kaiming初始化）的方法，它适用于ReLU激活函数
        nn.init.uniform_(m.weight, -bound, bound)       # 使用均匀分布初始化权重
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):                   # 检查模型中的层 m 是否是一个嵌入层
        nn.init.normal_(m.weight, mean=0.0, std=0.02)   # 使用正态分布初始化权重，均值为0.0，标准差为0.02
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):     # 检查模型中的层 m 是否是批量归一化层的一个实例
        nn.init.ones_(m.weight)                         # 对于批量归一化层，将权重（也称为缩放因子）初始化为1
        nn.init.zeros_(m.bias)                          # 将偏置初始化为0
    elif isinstance(m, nn.LayerNorm):                   # 检查模型中的层 m 是否是层归一化层的一个实例
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):          # 检查模型中的层 m 是否是多头注意力机制曾
        if m.in_proj_weight is not None:        # 检查多头注意力层是否有输入投影权重
            fan_in = m.embed_dim                # 计算权重初始化的“fan in”和“fan out”。在这里，它们都被设置为嵌入维度embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:           # 如果没有输入投影权重，那么分别初始化查询（q）、键（k）和值（v）的投影权重
            nn.init.xavier_uniform_(m.q_proj_weight)      # 使用Xavier均匀初始化方法初始化这些权重
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:          # 检查是否有输入投影偏置
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:         # 检查是否有输出投影偏置
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:                # 检查是否有键的偏置
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)    # 使用正态分布初始化，均值为0.0，标准差为0.02
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell)):                # 检查模型中的层 m 是否是长短期记忆网络或LSTM单元层类的一个实例
        for name, param in m.named_parameters():      # 遍历LSTM或LSTMCell层的所有参数，name是参数的名称，param是对应的参数张量
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):    # 将weight_ih权重张量分割成4个部分，每个部分对应一个门（输入门、遗忘门、单元状态、输出门）
                    nn.init.xavier_uniform_(ih)              # 使用 Xavier 均匀初始化方法初始化每个门的权重
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)           # 使用正交初始化方法初始化每个门的权重。正交初始化有助于保持梯度的规模，有助于避免梯度消失或爆炸。
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])    # 将遗忘门的偏置初始化为1,以允许信息在时间步之间流动
    elif isinstance(m, (nn.GRU, nn.GRUCell)):        # 检查模型中的层 m 是否是 nn.GRU 或 nn.GRUCell 类的一个实例
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):  # 这一行将weight_ih权重张量分割成3个部分，每个部分对应一个更新门、重置门和候选隐藏状态
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
