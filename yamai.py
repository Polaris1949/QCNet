import torch
from datamodules import ArgoverseV2DataModule

from dataclasses import dataclass
from typing import Any, Tuple, List, Optional
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

MAX_NUM_AGENTS = 135
MAX_DISTANCE = 200.0 # 196.14

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

    def radius(self, r: int):
        from torch_cluster import radius
        pos_x = self.position[:-1, :, :2].reshape(-1, 2)
        pos_y = self.position[-1, :, :2].reshape(-1, 2)
        edge_index = radius(pos_x, pos_y, r, max_num_neighbors=100)
        mask_x = self.valid_mask[:-1, :].reshape(-1)
        mask_y = self.valid_mask[-1, :].reshape(-1)
        edge_mask = mask_x[edge_index[1]] & mask_y[edge_index[0]]
        edge_index = edge_index[:, edge_mask]
        agent_ids = edge_index[1] // self.num_steps
        step_ids = edge_index[1] % self.num_steps
        distances = torch.norm(pos_x[edge_index[1]] - pos_y[edge_index[0]], p=2, dim=1)
        return agent_ids, step_ids, distances

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
    def from_batch(cls, data: Any) -> List["AstreaScene"]:
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
                agent=AstreaAgent(
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

def plot_density(y_samples):
    # Convert torch tensor to numpy array if needed
    if isinstance(y_samples, torch.Tensor):
        y_samples = y_samples.detach().cpu().numpy()

    # Ensure y_samples is 1D
    y_samples = np.ravel(y_samples)

    # Perform kernel density estimation
    kde = gaussian_kde(y_samples)

    # Create x values for the plot (0 to 200)
    x = np.linspace(0, 200, 1000)

    # Calculate density values
    density = kde(x)

    # Normalize density to ensure area under curve equals 1
    area = np.trapz(density, x)
    density = density / area

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, density, 'b-', label='Density')
    plt.fill_between(x, density, alpha=0.2)
    plt.xlabel('y values')
    plt.ylabel('Density')
    plt.title('Density Plot of y_samples')
    plt.xlim(0, 200)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    return x, density

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
    min_num_agents = 10000
    max_distance = 0.0
    max_dists = []
    dists = []

    for data in chain(train_dataloader, val_dataloader, test_dataloader):  # 遍历各个场景下车辆的数据集
        for scene in AstreaScene.from_batch(data):
            # print(scene.agent)
            min_num_agents = min(min_num_agents, scene.agent.num_nodes)
            max_num_agents = max(max_num_agents, scene.agent.num_nodes)
            agent_ids, step_ids, local_dists = scene.agent.radius(200)
            dists.append(local_dists)
            # print(f"Scenario {scene.scenario_id}, Agent {agent_id}, Step {step_id}, Distance {dist:.2f}")
            local_max_dist = torch.max(local_dists).item()
            max_dists.append(local_max_dist)
            print(f'{scene.scenario_id=}, {local_max_dist=:.2f}')
            max_distance = max(max_distance, local_max_dist)

    print(f'{min_num_agents=}')
    print(f'{max_num_agents=}')
    print(f'{max_distance=:.2f}')
    max_dists = torch.tensor(max_dists)
    print("Figure 1")
    plot_density(max_dists)
    dists = torch.cat(dists, dim=0).reshape(-1)
    print("Figure 2")
    plot_density(dists)
