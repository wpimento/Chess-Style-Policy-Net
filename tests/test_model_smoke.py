"""
small CPU-only smoke test for the two-head policy network

run from the project root:
    python -m unittest discover -s tests -v

this test uses random tensors and does NOT demonstrate chess-playing ability
"""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from move_vocab import MOVE_VOCAB_SIZE
from policy_network import PolicyNetwork


class PolicyNetworkSmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.network = PolicyNetwork(num_moves=MOVE_VOCAB_SIZE)
        self.board_batch = torch.randn(2, 18, 8, 8)
        self.context_batch = torch.randn(2, 15)

    def test_output_dimensions(self):
        self.network.eval()
        with torch.no_grad():
            policy_logits, time_logits = self.network(
                self.board_batch, self.context_batch
            )
        self.assertEqual(tuple(policy_logits.shape), (2, MOVE_VOCAB_SIZE))
        self.assertEqual(tuple(time_logits.shape), (2, 5))

    def test_multitask_loss_can_backpropagate(self):
        self.network.train()
        policy_logits, time_logits = self.network(
            self.board_batch, self.context_batch
        )
        policy_targets = torch.tensor([0, MOVE_VOCAB_SIZE - 1], dtype=torch.long)
        time_targets = torch.tensor([2, 0], dtype=torch.long)
        has_time_target = torch.tensor([True, False], dtype=torch.bool)

        policy_loss = torch.nn.functional.cross_entropy(
            policy_logits, policy_targets
        )
        per_sample_time_loss = torch.nn.functional.cross_entropy(
            time_logits, time_targets, reduction="none"
        )
        move_time_loss = per_sample_time_loss[has_time_target].mean()
        total_loss = policy_loss + 0.2 * move_time_loss
        self.assertTrue(torch.isfinite(total_loss).item())
        total_loss.backward()

        self.assertIsNotNone(self.network.policy_head[-1].weight.grad)
        self.assertIsNotNone(self.network.move_time_head[-1].weight.grad)


if __name__ == "__main__":
    unittest.main()