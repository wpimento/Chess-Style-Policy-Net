from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/position_dataset.parquet")

TRAIN_PATH = Path("data/processed/train_positions.parquet")
VALIDATION_PATH = Path("data/processed/validation_positions.parquet")
TEST_PATH = Path("data/processed/test_positions.parquet")

TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1

def split_position_dataset() -> None:
    pass

if __name__ == "__main__":
    split_position_dataset()