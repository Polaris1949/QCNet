from dataclasses import dataclass
import math
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import pandas as pd
import torch
from torch_cluster import radius_graph
from tqdm import tqdm
from datasets.yamai_common import *
from utils import angle_between_2d_vectors, wrap_angle
from datasets.argoverse_v2_dataset import ArgoverseV2Dataset, read_json_file, ArgoverseStaticMap, Polyline

MIKU_FORCE_PROCESS = False

MIKU_DIST_MIN = 1e6
MIKU_DIST_MAX = 0.0
MIKU_SPEED_MIN = 1e6
MIKU_SPEED_MAX = 0.0

MIKU_DIST_CLAMP_MAX = 200.0
MIKU_SPEED_CLAMP_MAX = 20.0

def clamp_number_ratio(input: torch.Tensor, max: float) -> torch.Tensor:
    return torch.clamp_max(input, max) / max

def clamp_angle_ratio(input: torch.Tensor) -> torch.Tensor:
    return (input + math.pi) / (math.pi * 2)

@dataclass
class MikuAgent:
    valid_mask: torch.Tensor # [92, 110]
    predict_mask: torch.Tensor # [92, 110]
    id: List[Optional[str]] # [92]
    type: torch.Tensor # [92]
    category: torch.Tensor # [92]
    position: torch.Tensor # [92, 110, 3]
    heading: torch.Tensor # [92, 110]
    velocity: torch.Tensor # [92, 110, 3]

    @property
    def num_nodes(self) -> int:
        return self.valid_mask.shape[0]

    @property
    def av_index(self) -> int:
        return self.num_nodes - 1

    @property
    def num_steps(self) -> int:
        return self.valid_mask.shape[1]

    def exists(self, agent_idx: int, step_idx: int):
        return self.valid_mask[agent_idx][step_idx]

    def stat(self, agent_idx: int, step_idx: int):
        pos_t = self.position[agent_idx][step_idx][:2]
        theta_t = self.heading[agent_idx][step_idx].unsqueeze(0)
        vel_t = self.velocity[agent_idx][step_idx][:2]
        return pos_t, theta_t, vel_t

    def __repr__(self) -> str:
        return (
            f"AstreaAgent(num_nodes={self.num_nodes}, num_steps={self.num_steps}, "
            f"valid_mask={self.valid_mask.shape}, predict_mask={self.predict_mask.shape}, "
            f"id={len(self.id)}, type={self.type.shape}, category={self.category.shape}, "
            f"position={self.position.shape}, heading={self.heading.shape}, velocity={self.velocity.shape})"
        )

    def to_graphs(self) -> List[YamaiGraph]:
        # 智能体位置 [A, T, 2]
        pos_a = self.position[:, :NUM_HISTORICAL_STEPS, :INPUT_DIM].contiguous()
        # 智能体位移向量 [A, T, 2]
        motion_vector_a = torch.cat([pos_a.new_zeros(self.num_nodes, 1, INPUT_DIM),
                                     pos_a[:, 1:] - pos_a[:, :-1]], dim=1)
        # 智能体朝向角 [A, T]
        head_a = self.heading[:, :NUM_HISTORICAL_STEPS].contiguous()
        # 智能体朝向向量 [A, T, 2]
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        # 智能体速度向量 [A, T, 2]
        vel = self.velocity[:, :NUM_HISTORICAL_STEPS, :INPUT_DIM].contiguous()

        # global MIKU_DIST_MAX, MIKU_DIST_MIN, MIKU_SPEED_MAX, MIKU_SPEED_MIN
        # 智能体移动距离 [A, T]
        motion_dist_a = torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1)
        motion_dist_a = clamp_number_ratio(motion_dist_a, MIKU_DIST_CLAMP_MAX)
        # MIKU_DIST_MAX = max(MIKU_DIST_MAX, torch.max(motion_dist_a).item())
        # MIKU_DIST_MIN = min(MIKU_DIST_MIN, torch.min(motion_dist_a).item())
        # 智能体朝向向量和位移向量之间的夹角 [A, T]
        angle_head_motion_a = angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2])
        angle_head_motion_a = clamp_angle_ratio(angle_head_motion_a)
        # 智能体速率 [A, T]
        speed_a = torch.norm(vel[:, :, :2], p=2, dim=-1)
        speed_a = clamp_number_ratio(speed_a, MIKU_SPEED_CLAMP_MAX)
        # MIKU_SPEED_MAX = max(MIKU_SPEED_MAX, torch.max(speed_a).item())
        # MIKU_SPEED_MIN = min(MIKU_SPEED_MIN, torch.min(speed_a).item())
        # 智能体朝向向量和速度向量之间的夹角 [A, T]
        angle_head_velocity_a = angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel[:, :, :2])
        angle_head_velocity_a = clamp_angle_ratio(angle_head_velocity_a)
        # 智能体特征 [A, T, 4]
        x_a = torch.stack([motion_dist_a, angle_head_motion_a, speed_a, angle_head_velocity_a], dim=-1)

        # 智能体数目
        num_nodes = self.num_nodes
        # 生成时间跨度内的所有顶点
        num_nodes_t = num_nodes * TIME_SPAN
        # 生成完全图边集
        node_full_graph = torch.tensor([(i, j) for i in range(num_nodes_t) for j in range(num_nodes_t)]).transpose(0, 1)
        yamai_graphs: List[YamaiGraph] = []

        for time_beg in range(0, NUM_HISTORICAL_STEPS, TIME_SPAN):
            time_end = time_beg + TIME_SPAN
            # 当前时间跨度内的特征 [A*Ts, 4]
            x_a_t = x_a[:, time_beg:time_end, :].reshape(num_nodes_t, 4)
            pos_a_t = pos_a[:, time_beg:time_end, :].reshape(num_nodes_t, 2)
            head_a_t = head_a[:, time_beg:time_end].reshape(num_nodes_t)
            head_vector_a_t = head_vector_a[:, time_beg:time_end, :].reshape(num_nodes_t, 2)
            rel_pos_a2a = pos_a_t[node_full_graph[0]] - pos_a_t[node_full_graph[1]]
            # 智能体两两之间的夹角
            rel_head_a2a = wrap_angle(head_a_t[node_full_graph[0]] - head_a_t[node_full_graph[1]])
            rel_head_a2a = clamp_angle_ratio(rel_head_a2a)
            # 智能体两两之间的相对距离
            rel_dist_a2a = torch.norm(rel_pos_a2a, p=2, dim=-1)
            rel_dist_a2a = clamp_number_ratio(rel_dist_a2a, MIKU_DIST_CLAMP_MAX)
            # MIKU_DIST_MAX = max(MIKU_DIST_MAX, torch.max(rel_dist_a2a).item())
            # MIKU_DIST_MIN = min(MIKU_DIST_MIN, torch.min(rel_dist_a2a).item())
            # 智能体两两之间的碰撞转角
            rel_coll_a2a = angle_between_2d_vectors(ctr_vector=head_vector_a_t[node_full_graph[1]], nbr_vector=rel_pos_a2a)
            rel_coll_a2a = clamp_angle_ratio(rel_coll_a2a)
            # 智能体相关性 [A*Ts, A*Ts*3]
            r_a2a = torch.stack([rel_dist_a2a, rel_coll_a2a, rel_head_a2a], dim=-1).reshape(num_nodes_t, num_nodes_t * 3)
            # 输出特征 [A*Ts, 4+A*Ts*3]
            x_yamai = torch.cat([x_a_t, r_a2a], dim=1)
            edge_index_a2a = radius_graph(x=pos_a_t, r=A2A_RADIUS, loop=False, max_num_neighbors=300)
            yamai_graphs.append(YamaiGraph(x=x_yamai, edge_index=edge_index_a2a))

        return yamai_graphs

@dataclass
class MikuScene:
    scenario_id: str # uuid
    city: str # name
    agent: MikuAgent

    @classmethod
    def from_av2(cls, data: Any) -> "MikuScene":
        _agent = data['agent']
        _num_nodes = _agent['num_nodes']
        _av_index = int(_agent['av_index'])
        assert _num_nodes == _av_index + 1

        return cls(
            scenario_id=data['scenario_id'],
            city=data['city'],
            agent=MikuAgent(
                valid_mask=_agent['valid_mask'],
                predict_mask=_agent['predict_mask'],
                id=_agent['id'],
                type=_agent['type'],
                category=_agent['category'],
                position=_agent['position'],
                heading=_agent['heading'],
                velocity=_agent['velocity'],
            ),
        )

    @classmethod
    def from_batch(cls, data: Any) -> List["MikuScene"]:
        batch_size = len(data['scenario_id'])
        o = []

        for i in range(batch_size):
            agent = data['agent']
            ptr_beg = agent['ptr'][i]
            ptr_end = agent['ptr'][i + 1]
            num_nodes = ptr_end - ptr_beg
            av_index = int(agent['av_index'][i])
            assert num_nodes == av_index + 1

            o.append(cls(
                scenario_id=data['scenario_id'][i],
                city=data['city'][i],
                agent=MikuAgent(
                    valid_mask=agent['valid_mask'][ptr_beg:ptr_end],
                    predict_mask=agent['predict_mask'][ptr_beg:ptr_end],
                    id=agent['id'][i],
                    type=agent['type'][ptr_beg:ptr_end],
                    category=agent['category'][ptr_beg:ptr_end],
                    position=agent['position'][ptr_beg:ptr_end],
                    heading=agent['heading'][ptr_beg:ptr_end],
                    velocity=agent['velocity'][ptr_beg:ptr_end],
                )
            ))

        return o

    def to_yamai(self) -> Yamai:
        return Yamai(graphs=self.agent.to_graphs())

class MikuDataset(ArgoverseV2Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 raw_dir: Optional[str] = None,
                 processed_dir: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 dim: int = 3,           # 数据维度
                 num_historical_steps: int = 50,
                 num_future_steps: int = 60,
                 predict_unseen_agents: bool = False,                #  预测未见智能体
                 vector_repr: bool = True) -> None:                  #  向量表示
        if processed_dir is None:
            processed_dir = os.path.join(root, split, "yamai")
        super().__init__(root, split, raw_dir, processed_dir, transform, dim,
                         num_historical_steps, num_future_steps, predict_unseen_agents, vector_repr)

    def _process(self) -> None:
        if MIKU_FORCE_PROCESS:
            self._processed_file_names = []
        super()._process()

    def process(self) -> None:
        for raw_file_name in tqdm(self.raw_file_names):               # tqdm是一个进度条库，用于在控制台显示进度
            df = pd.read_parquet(os.path.join(self.raw_dir, raw_file_name, f'scenario_{raw_file_name}.parquet'))     # 用pandas库读取一个Parquet文件
            map_dir = Path(self.raw_dir) / raw_file_name              # 使用pathlib库创建一个路径对象，指向包含地图数据的目录
            map_path = sorted(map_dir.glob('log_map_archive_*.json'))[0]   # 在map_dir目录中查找所有以log_map_archive_开头的 JSON 文件，并选择第一个文件作为地图数据文件
            map_data = read_json_file(map_path)              # 读取地图数据json文件，并将内容存储在map_data变量中
            centerlines = {lane_segment['id']: Polyline.from_json_data(lane_segment['centerline'])
                           for lane_segment in map_data['lane_segments'].values()}              # 从地图数据中提取车道线信息，并创建一个车道线id到Polyline对象的映射
            map_api = ArgoverseStaticMap.from_json(map_path)               # 使用ArgoverseStaticMap类从地图数据文件创建一个地图 API 对象
            data = dict()                 # 创建一个空字典，用于存储处理后的数据
            data['scenario_id'] = self.get_scenario_id(df)               # 调用self.get_scenario_id，从数据帧df中获取场景id，并将其添加到data字典中
            data['city'] = self.get_city(df)
            data['agent'] = self.get_agent_features(df)
            data.update(self.get_map_features(map_api, centerlines))        # 调用self.get_map_features方法，获取地图特征，并将结果更新到data字典中
            yamai = MikuScene.from_av2(data).to_yamai()
            with open(os.path.join(self.processed_dir, f'{raw_file_name}.pkl'), 'wb') as handle:   # 打开一个raw_file_name文件，并使用pickle序列化数据
                pickle.dump(yamai.dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)             #使用HIGHEST_PROTOCOL可以确保代码使用的是当前Python版本支持的最新协议
