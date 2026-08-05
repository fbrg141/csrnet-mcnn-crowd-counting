#!/usr/bin/env python3
"""Download the ShanghaiTech crowd counting dataset."""

from pathlib import Path

import kagglehub

DATASET = "tthien/shanghaitech"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{DATASET}'...")
    path = kagglehub.dataset_download(DATASET)
    print(f"Downloaded to: {path}")

    # Copy ShanghaiTech directory into data/raw/
    src = Path(path) / "ShanghaiTech"
    dst = RAW_DIR / "ShanghaiTech"

    if dst.exists():
        print(f"Removing existing: {dst}")
        import shutil
        shutil.rmtree(dst)

    print(f"Copying to: {dst}")
    import shutil
    shutil.copytree(src, dst)

    # Quick verification
    part_a_train = len(list(dst.glob("part_A/train_data/images/*.jpg")))
    part_a_test = len(list(dst.glob("part_A/test_data/images/*.jpg")))
    part_b_train = len(list(dst.glob("part_B/train_data/images/*.jpg")))
    part_b_test = len(list(dst.glob("part_B/test_data/images/*.jpg")))

    print()
    print("Dataset ready:")
    print(f"  Part A — {part_a_train} train, {part_a_test} test")
    print(f"  Part B — {part_b_train} train, {part_b_test} test")


if __name__ == "__main__":
    main()
