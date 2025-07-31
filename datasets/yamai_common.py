from dataclasses import dataclass, asdict
from typing import Any, Dict, List
import torch
from torch_geometric.data import Data

NUM_MAX_AGENTS = 135
NUM_HISTORICAL_STEPS = 50
TIME_SPAN = 10
INPUT_DIM = 2
HIDDEN_DIM = 128
A2A_RADIUS = 50
NUM_NODES = NUM_MAX_AGENTS * TIME_SPAN
NODE_FULL_GRAPH = torch.tensor([(i, j) for i in range(NUM_NODES) for j in range(NUM_NODES)]).transpose(0, 1)

@dataclass
class YamaiGraph:
    x: torch.Tensor
    edge_index: torch.Tensor

    @classmethod
    def from_data(cls, data: Data) -> "YamaiGraph":
        return cls(x=data.x, edge_index=data.edge_index) # type: ignore

    @classmethod
    def from_batch(cls, data: Dict[str, Any]) -> "YamaiGraph":
        return cls(x=data['x'][0], edge_index=data['edge_index'][0])

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]

    @property
    def num_features(self) -> int:
        return self.x.shape[1]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def __repr__(self) -> str:
        return f"YamaiGraph(x={self.x.shape}, edge_index={self.edge_index.shape})"

@dataclass
class Yamai:
    graphs: List[YamaiGraph]

    @classmethod
    def from_batch(cls, data: Dict[str, Any]) -> "Yamai":
        return Yamai(graphs=[YamaiGraph.from_batch(i) for i in data['graphs']])

    def dict(self) -> Dict[str, Any]:
        return asdict(self)
