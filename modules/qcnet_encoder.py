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
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from modules.qcnet_agent_encoder import QCNetAgentEncoder
from modules.qcnet_map_encoder import QCNetMapEncoder



class QCNetEncoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,       # 地图多边形之间的交互半径
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_map_layers: int,        # 地图编码器中的层数
                 num_agent_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(QCNetEncoder, self).__init__()
        self.map_encoder = QCNetMapEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.agent_encoder = QCNetAgentEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:        # 异构数据
        map_enc = self.map_encoder(data) # 调用地图编码器：调用map_encoder来处理输入数据中的地图相关信息
        agent_enc = self.agent_encoder(data, map_enc) # 调用智能体编码器：调用agent_encoder来处理输入数据中的智能体相关信息
        return {**map_enc, **agent_enc}  # 返回一个字典，其中包含了地图编码和智能体编码的结果
