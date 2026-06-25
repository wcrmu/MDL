import math
import unittest

import torch

from mdl import MDLConfig, MDLModel, multitask_bce_loss, qauc
from mdl.training import binary_auc
from mdl.training import make_synthetic_batch


class MDLSmokeTest(unittest.TestCase):
    def make_config(self) -> MDLConfig:
        return MDLConfig(
            num_feature_tokens=3,
            scenario_context_dim=6,
            task_context_dim=7,
            num_scenarios=2,
            num_tasks=3,
            token_dim=24,
            num_layers=2,
            num_heads=4,
            ffn_hidden_dim=24,
            feature_backbone="rankmixer",
        )

    def test_forward_shape_and_attention(self) -> None:
        config = self.make_config()
        model = MDLModel(config)
        batch = make_synthetic_batch(config, batch_size=8)

        output = model(
            batch.feature_tokens,
            batch.scenario_context,
            batch.task_context,
            batch.scenario_mask,
            return_attention=True,
        )

        self.assertEqual(output["logits"].shape, (8, config.num_tasks))
        self.assertEqual(len(output["attentions"]), config.num_layers)
        self.assertEqual(
            output["attentions"][0]["task_feature"].shape,
            (8, config.num_heads, config.num_tasks, config.num_feature_tokens),
        )

    def test_backward_step(self) -> None:
        config = self.make_config()
        model = MDLModel(config)
        batch = make_synthetic_batch(config, batch_size=8)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

        output = model(
            batch.feature_tokens,
            batch.scenario_context,
            batch.task_context,
            batch.scenario_mask,
        )
        loss = multitask_bce_loss(output["logits"], batch.labels, batch.label_mask)
        loss.backward()
        optimizer.step()

        self.assertTrue(math.isfinite(loss.item()))

    def test_masked_loss_ignores_missing_labels(self) -> None:
        logits = torch.tensor([[0.0, 10.0], [0.0, -10.0]])
        labels = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        masked = multitask_bce_loss(logits, labels, mask)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, 0], labels[:, 0]
        )

        self.assertAlmostEqual(masked.item(), expected.item(), places=6)

    def test_qauc_skips_single_class_groups(self) -> None:
        result = qauc(
            labels=[0, 1, 1, 1],
            scores=[0.1, 0.9, 0.2, 0.3],
            query_ids=["a", "a", "b", "b"],
        )

        self.assertEqual(result.valid_groups, 1)
        self.assertEqual(result.skipped_groups, 1)
        self.assertAlmostEqual(result.qauc, 1.0)

    def test_binary_auc_ties(self) -> None:
        auc = binary_auc(labels=[0, 1], scores=[0.5, 0.5])

        self.assertAlmostEqual(auc, 0.5)


if __name__ == "__main__":
    unittest.main()

