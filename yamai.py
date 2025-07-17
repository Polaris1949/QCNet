import torch
from datamodules import ArgoverseV2DataModule

from dataclasses import dataclass
from typing import Any, Tuple, List, Optional
from itertools import chain

@dataclass
class AstreaAgent:
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

@dataclass
class AstreaScene:
    scenario_id: str # uuid
    city: str # name
    agent: AstreaAgent

    @classmethod
    def parse(cls, data: Any) -> "AstreaScene":
        _agent = data['agent']
        _num_nodes = _agent['num_nodes']
        _av_index = int(_agent['av_index'][0])
        assert _num_nodes == _av_index + 1

        return cls(
            scenario_id=data['scenario_id'][0],
            city=data['city'][0],
            agent=AstreaAgent(
                valid_mask=_agent['valid_mask'],
                predict_mask=_agent['predict_mask'],
                id=_agent['id'][0],
                type=_agent['type'],
                category=_agent['category'],
                position=_agent['position'],
                heading=_agent['heading'],
                velocity=_agent['velocity'],
            ),
        )


if __name__ == '__main__':
    # 准备数据集Av2
    root = './data_av2'  # 读取数据集路径
    train_batch_size = 1
    val_batch_size = 1
    test_batch_size = 1
    datamodule = ArgoverseV2DataModule(root=root, train_batch_size=train_batch_size,
                                       val_batch_size=val_batch_size, test_batch_size=test_batch_size)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    max_num_agents = 0

    for data in chain(train_dataloader, val_dataloader, test_dataloader):  # 遍历各个场景下车辆的数据集
        scene = AstreaScene.parse(data)
        max_num_agents = max(max_num_agents, scene.agent.num_nodes)

    print(f'{max_num_agents=}')
