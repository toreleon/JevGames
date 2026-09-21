from __future__ import annotations

import importlib.util
import unittest


HAS_QWEN_RUNTIME = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"))


@unittest.skipUnless(HAS_QWEN_RUNTIME, "Qwen adapter dependencies are optional")
class QwenAdapterTests(unittest.TestCase):
    def test_causal_decision_head_backpropagates_from_final_token(self) -> None:
        import torch
        from transformers import Qwen3Config, Qwen3Model

        from jevgames.models.qwen import QwenDecisionModel

        torch.manual_seed(7)
        backbone = Qwen3Model(Qwen3Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            attention_dropout=0.0,
        ))
        model = QwenDecisionModel.build(backbone)
        input_ids = torch.tensor([[1, 2, 3, 4], [1, 2, 8, 9]])
        scores = model(input_ids, torch.ones_like(input_ids))
        scores.sum().backward()
        self.assertEqual(tuple(scores.shape), (2,))
        self.assertTrue(bool(torch.isfinite(scores).all()))
        self.assertGreater(float(model.scorer[-1].weight.grad.norm()), 0.0)
        self.assertGreater(float(model.backbone.embed_tokens.weight.grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
