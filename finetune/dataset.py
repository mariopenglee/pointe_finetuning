import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Optional, Tuple

class PointCloudTextDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        pc_dir: Path,
        preload: bool = True,
        device: Optional[torch.device] = None,
    ):
        df = pd.read_csv(csv_path)
        self.stems   = [Path(f).stem for f in df["filename"].tolist()]
        self.prompts = df["prompt"].tolist()
        self.pc_dir  = Path(pc_dir)
        self.device  = device or torch.device("cpu")

        if preload:
            self._pcs: List[torch.Tensor] = []
            for stem in self.stems:
                fp = self.pc_dir / f"{stem}_pc.pt"
                data = torch.load(fp, map_location=self.device)
                pc = data["pointcloud"]
                self._pcs.append(pc.float())
        else:
            self._pcs = None

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        prompt = self.prompts[idx]
        if self._pcs is not None:
            pc = self._pcs[idx]
        else:
            stem = self.stems[idx]
            fp = self.pc_dir / f"{stem}_pc.pt"
            data = torch.load(fp, map_location=self.device)
            pc   = data["pointcloud"].float()
        return prompt, pc