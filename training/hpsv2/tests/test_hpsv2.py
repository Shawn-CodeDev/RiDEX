import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from ridex.config import Config
from ridex.data import epoch_plan, load_dataset
from ridex.reward_protocol import PROTOCOL, decode_batch, encode_batch
from ridex.rollout import directional_loss, pair_credit
from ridex.scoring import canonical_scorer, component_at, component_metrics
from ridex.train import policy_update


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTest(unittest.TestCase):
    def test_both_published_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for split in ("train", "test"):
                (directory / f"{split}.txt").write_text("\n".join(f"Prompt {i}" for i in range(60)))
            for filename, images, pairs, prompts_per_rank in (
                ("hpsv2.json", 576, 64, 1), ("diffusionnft_hpsv2.json", 1152, 0, 3)
            ):
                data = json.loads((ROOT / "configs" / filename).read_text())
                data.update(dataset=str(directory), model="model", reward_models="reward")
                config = Config(**data)
                config.validate(16)
                budget = config.budget()
                self.assertEqual((budget["total_images"], budget["pair_images"]), (images, pairs))
                self.assertEqual(budget["direction_backward_calls"], pairs // 2)
                self.assertEqual(config.phase_end_update(1), 1)
                plan = epoch_plan(config, load_dataset(config, "train"), 0)
                self.assertTrue(all(len(plan[rank::16]) == prompts_per_rank for rank in range(16)))

    def test_nft_update_uses_only_nft_gradient(self):
        params = [torch.nn.Parameter(torch.tensor([0.2, -0.3]))]
        ref = [torch.nn.Parameter(params[0].detach().clone())]
        optimizer = torch.optim.AdamW(params, lr=3e-4, eps=1e-8, weight_decay=1e-4)
        expected = torch.optim.AdamW(ref, lr=3e-4, eps=1e-8, weight_decay=1e-4)
        grad = torch.tensor([-0.3, 0.4])
        ref[0].grad = grad.clone()
        expected.step()
        _, loss, stats = policy_update(None, None, params, optimizer, [grad], [], [], 0, None, 16)
        torch.testing.assert_close(params[0], ref[0], rtol=0, atol=0)
        self.assertEqual((loss, stats), (0., {}))

    def test_h200_submit_and_service_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(ROOT / "submit.sh", root / "submit.sh")
            (root / "assets").mkdir()
            for relative in ("assets/train.txt", "assets/test.txt",
                "assets/HPSv2/HPS_v2.1_compressed.pt",
                "assets/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"):
                path = root / relative
                path.parent.mkdir(exist_ok=True)
                path.write_text("test fixture\n")
            (root / "site.env").write_text(
                f'export RIDEX_ROOT="{root}"\nexport PYTHON="{sys.executable}"\n'
                'export TRAIN_NODES=2 GPUS_PER_NODE=8 H200_GRES_TYPE=test_h200\n'
                'export TRAIN_PARTITION=training SERVICE_PARTITION=scorers\n'
                'export SD3_MODEL="$RIDEX_ROOT/assets" PROMPT_DATASET="$RIDEX_ROOT/assets" '
                'REWARD_MODEL_ROOT="$RIDEX_ROOT/assets" HPSV2_REWARD_URLS=\n'
            )
            sbatch = root / "sbatch"
            sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$RIDEX_ROOT/calls.txt"\nprintf "1234\\n"\n')
            sbatch.chmod(0o755)
            scancel = root / "scancel"
            scancel.write_text('#!/bin/bash\nexit 0\n')
            scancel.chmod(0o755)
            env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}")
            for profile, config in (("ridex", "hpsv2.json"), ("diffusionnft", "diffusionnft_hpsv2.json")):
                result = subprocess.run(["bash", "submit.sh", "--submit", profile, "--stop-after-updates", "1"],
                    cwd=root, env=env, check=True, capture_output=True, text=True)
                self.assertIn("train=1234", result.stdout)
                train, scorer, cleanup = (root / "calls.txt").read_text().splitlines()[-3:]
                self.assertIn("--gres=gpu:test_h200:8", train)
                self.assertIn("--nodes=2", train)
                self.assertIn(f"configs/{config} --stop-after-updates 1", train)
                self.assertIn("--dependency=after:1234", scorer)
                self.assertIn("--dependency=afterany:1234", cleanup)

    def test_reference_budget_and_seed_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            prompts = "\n".join(f"Prompt {i}" for i in range(20))
            for split in ("train", "test"):
                (directory / f"{split}.txt").write_text(prompts, encoding="utf-8")
            config = Config(model="model", reward_models="reward", dataset=str(directory))
            config.validate(16)
            self.assertEqual(config.budget(), dict(clean_images=512, pair_images=64,
                total_images=576, direction_backward_calls=32, canonical_scorer_images=576))
            records = load_dataset(config, "train")
            plan = epoch_plan(config, records, 0)
            self.assertEqual(plan, epoch_plan(config, records, 0))
            self.assertNotEqual(plan, epoch_plan(config, records, 1))
            self.assertEqual(len({group["prompt_id"] for group in plan}), 16)
            self.assertTrue(all(len(group["seeds"]) == 32 for group in plan))

    def test_signed_pair_credit(self):
        credit = pair_credit(.32, .30, .005)
        velocity = torch.tensor([0., 0.], requires_grad=True)
        loss = directional_loss(velocity, velocity.detach(), torch.tensor([1., 0.]), torch.tensor(2.), credit)
        loss.backward()
        torch.testing.assert_close(velocity.grad, torch.tensor([-1., 0.]))
        self.assertEqual(pair_credit(.30, .30, .005), 0.)
        self.assertLess(pair_credit(.30, .32, .005), 0.)

    def test_float32_transport_and_validation(self):
        images = np.random.default_rng(42).random((2, 3, 7, 11), dtype=np.float32)
        prompts = ["an ink drawing", "a lighthouse"]
        body = encode_batch(images, prompts)
        recovered, texts = decode_batch(body)
        np.testing.assert_array_equal(images, recovered)
        self.assertEqual(prompts, texts)
        for malformed in (body[:-1], b"invalid", encode_batch(images, prompts[:1])):
            with self.assertRaises(ValueError):
                decode_batch(malformed)

    @mock.patch.dict(os.environ, {"HPSV2_REWARD_URLS": "http://localhost:18300", "HPSV2_REWARD_BATCH_SIZE": "1"})
    @mock.patch("ridex.remote.requests.Session")
    def test_remote_canonical_and_display_units(self, session_class):
        session = session_class.return_value
        response = session.post.return_value
        response.status_code = 200
        response.json.side_effect = [{"scores": [.30], "reward_protocol": PROTOCOL},
                                     {"scores": [.32], "reward_protocol": PROTOCOL}]
        values, info = canonical_scorer()(torch.zeros(2, 3, 5, 5), ["a", "b"], [{}, {}])
        self.assertEqual(values, [.30, .32])
        rows = [{"components": component_at(info, i)} for i in range(2)]
        self.assertEqual(component_metrics(rows, "eval_reward"),
            {"eval_reward_hpsv2": 31., "eval_reward/hpsv2_canonical": .31})
        self.assertEqual(session.post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
