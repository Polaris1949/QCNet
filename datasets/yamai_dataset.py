# Below is the REAL generated dataset, Yamai.
from torch.utils.data import Dataset as EasyDataset
from typing import Dict, Any
import os
import pickle

class YamaiDataset(EasyDataset[Dict[str, Any]]):
    def __init__(
        self,
        root: str,
        split: str,
        yamai_dir: str = "yamai",
        scene_num_graphs: int = 5,
    ) -> None:
        super().__init__()
        self.paths = [ent.path for ent in os.scandir(os.path.join(root, split, yamai_dir))]
        self.scene_num_graphs = scene_num_graphs

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_id, graph_id = divmod(idx, self.scene_num_graphs)
        with open(self.paths[file_id], 'rb') as fp:
            return pickle.load(fp)['graphs'][graph_id]

    def __len__(self) -> int:
        return len(self.paths) * self.scene_num_graphs
