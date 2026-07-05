"""Dataset loader placeholder for crowd counting.

This file will be adapted after we inspect the real annotation format.
"""

from pathlib import Path


class CrowdCountingDataset:
    """Minimal placeholder dataset class.

    Replace this skeleton after inspecting:
    - image directory layout
    - annotation file format
    - train/val/test splits
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int):
        raise NotImplementedError("Implement after dataset inspection.")
