
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chess_position_dataset import ChessPositionDataset
from move_vocab import MOVE_VOCAB_SIZE
from policy_network import PolicyNetwork


def main():
    dataset = ChessPositionDataset(
        "data/processed/train_positions.parquet"
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False
    )

    board, context, move_target, time_target, time_mask = next(iter(loader))

    model = PolicyNetwork(MOVE_VOCAB_SIZE)
    model.eval()

    with torch.no_grad():
        policy_logits, time_logits = model(board, context)

    print("Board:", board.shape)
    print("Context:", context.shape)
    print("Policy output:", policy_logits.shape)
    print("Move-time output:", time_logits.shape)

    assert policy_logits.shape == (board.shape[0], MOVE_VOCAB_SIZE)
    assert time_logits.shape == (board.shape[0], 5)

    print("Dataset forward-pass test passed!")


if __name__ == "__main__":
    main()
