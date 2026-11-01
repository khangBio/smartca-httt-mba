"""Local checks: python -m unittest discover -s kaggle -p 'test_*.py' -v"""
import importlib.util
import json
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import train_mba_kaggle as mba
import train_mba_a_kaggle as mba_a
import mba_a_kaggle_cell as mba_cell


class MBATests(unittest.TestCase):
    def test_notebook_config_ignores_kernel_argv(self):
        def parse_only(argv):
            return mba_cell.parse_args(argv)
        with patch.object(sys, "argv", ["launcher.py", "-f", "kernel.json", "--bad-flag"]):
            with patch.object(mba_cell, "main", side_effect=parse_only):
                with patch("builtins.print"):
                    args = mba_cell.run_notebook({
                        "device": "cpu", "dataset": "taobao", "epochs": 2,
                        "variants": ["B0", "B3"], "prepare_only": True,
                    })
        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.dataset, "taobao")
        self.assertEqual(args.variants, ["B0", "B3"])
        self.assertTrue(args.prepare_only)
        self.assertFalse(args.force)
        self.assertEqual(mba_cell.CONFIG["variants"], ["B3"])
        with self.assertRaises(ValueError):
            mba_cell.run_notebook({"epohs": 2})

    def test_jupyter_kernel_arguments(self):
        for module in (mba, mba_a):
            with self.subTest(module=module.__name__):
                with patch.object(sys, "argv", [
                    "colab_kernel_launcher.py", "-f", "/root/jupyter/kernel-test.json",
                    "--device", "cpu", "--dataset", "taobao", "--epochs", "3"
                ]):
                    args = module.parse_args()
                    self.assertEqual(args.dataset, "taobao")
                    self.assertEqual(args.epochs, 3)
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        module.parse_args(["--epohs", "3"])
                    self.assertEqual(error.exception.code, 2)
                self.assertEqual(module.parse_args(["--device", "cpu"]).variants,
                                 ["B3"] if module is mba_a else ["B0", "B3"])

    def test_original_loss_and_gradient_parity(self):
        """Generalized batched objective must equal the original pos/neg losses."""
        source = Path(__file__).resolve().parents[1] / "loss_func.py"
        spec = importlib.util.spec_from_file_location("original_loss", source)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        torch.manual_seed(5)
        for phase in ("DN", "DP"):
            for purchased in (True, False):
                logits = torch.randn(12, requires_grad=True)
                h = [torch.randn(12, requires_grad=True) for _ in range(4)]
                tb, tc = torch.randn(12), torch.randn(12)
                click_labels = torch.tensor([True] * 6 + [False] * 6)
                buy_labels = click_labels if purchased else torch.zeros_like(click_labels)
                got = (3 * (mba.bernoulli_kl(tb, logits) + mba.bernoulli_kl(tc, logits))
                       - mba.observation_log_likelihood(*h[:2], logits, buy_labels, phase, 7, 9)
                       - mba.observation_log_likelihood(*h[2:], logits, click_labels, phase, 7, 9))
                got = got[:6].mean() + got[6:].mean()
                expected = 0
                for labels, hs in ((buy_labels, h[:2]), (click_labels, h[2:])):
                    for start, end in ((0, 6), (6, 12)):
                        fun = (original.denoise_positive_loss if labels[start]
                               else original.denoise_negative_loss)
                        expected -= fun(hs[0][start:end], hs[1][start:end], logits[start:end],
                                        C=7 if labels[start] else 9,
                                        denoise_type=phase).mean()
                for teacher in (tb, tc):
                    terms = 3 * original.KL_loss(teacher, logits)
                    expected += terms[:6].mean() + terms[6:].mean()
                self.assertTrue(torch.allclose(got, expected, atol=2e-5))
                grad_got = torch.autograd.grad(got, logits, retain_graph=True)[0]
                grad_expected = torch.autograd.grad(expected, logits)[0]
                self.assertTrue(torch.allclose(grad_got, grad_expected, atol=1e-6))

    def test_sampler_and_split(self):
        pairs = np.array([[0, 0], [0, 1], [0, 2], [1, 1]], dtype=np.int64)
        train, val = mba.split_by_user(pairs, 0.2, 42)
        self.assertEqual(len(train), 3)
        self.assertEqual(len(val), 1)
        self.assertFalse(set(map(tuple, train)) & set(map(tuple, val)))
        self.assertEqual(set(train[:, 0]), {0, 1})
        sampler = mba.NegativeSampler(pairs, 5)
        users = np.tile([0, 1], 100)
        sampled = sampler.sample(users, np.random.default_rng(8))
        self.assertFalse(mba.contains(sampler.keys, users * 5 + sampled).any())

    def test_support_and_kl_gradients(self):
        class Counts:
            buy_counts = np.array([0, 1, 3, 10])
            click_counts = np.array([0, 1, 8, 30])
        buy, click = mba.support_weights(Counts(), 1, .05, "cpu")
        self.assertTrue(torch.allclose(buy + click, torch.full((4,), 2.0)))
        self.assertTrue((buy > 0).all() and (click > 0).all())
        teacher = torch.randn(4, requires_grad=True)
        target = torch.randn(4, requires_grad=True)
        mba.bernoulli_kl(teacher, target).sum().backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(target.grad).all())

    def test_ranking_metrics_and_mask(self):
        args = mba.parse_args(["--device", "cpu", "--top-k", "1", "2"])
        model = mba.MF(1, 3, 1)
        with torch.no_grad():
            model.user_embedding.weight.fill_(1)
            model.item_embedding.weight.copy_(torch.tensor([[3.], [2.], [1.]]))
        # item 0 masked; item 2 is at rank 2, so R@1=0, R@2=1.
        result = mba.evaluate(model, np.array([[0, 2]]), np.array([[0, 0]]), args)
        self.assertEqual(result["recall@1"], 0)
        self.assertEqual(result["recall@2"], 1)
        self.assertAlmostEqual(result["ndcg@2"], 1 / np.log2(3))

    def test_end_to_end_all_variants_and_cache(self):
        # Keep tiny regression outputs in workspace tmp for inspection. Windows
        # indexing/sandbox scanners can otherwise hold CSV handles during cleanup.
        workspace_tmp = Path(__file__).resolve().parents[1] / "tmp"
        workspace_tmp.mkdir(exist_ok=True)
        with nullcontext(tempfile.mkdtemp(prefix="kaggle_check_", dir=workspace_tmp)) as temporary:
            root = Path(temporary)
            buy, click, test = [], [], []
            for user in range(8):
                items = [(user * 4 + i) % 32 for i in range(6)]
                buy.extend((user, item) for item in items[:5])
                test.append((user, items[5]))
                click.extend((user, (user * 4 + i) % 32) for i in range(16))
            for name, pairs in (("buy_train", buy), ("pv_train", click), ("buy_test", test)):
                pd.DataFrame(pairs, columns=["uid", "sid"]).to_csv(
                    root / f"beibei_{name}.csv", sep="\t", index=False)
            command = [sys.executable, str(Path(mba.__file__)), "--data-root", str(root),
                       "--output", str(root / "results"), "--epochs", "1",
                       "--pretrain-epochs", "1", "--patience", "1", "--dim", "4",
                       "--batch-size", "16", "--eval-batch-size", "2", "--device", "cpu",
                       "--top-k", "3", "5", "--variants", "B0", "B1", "B2", "B3"]
            run = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            paths = list((root / "results").rglob("result.json"))
            self.assertEqual(len(paths), 4)
            for path in paths:
                result = json.loads(path.read_text())
                self.assertEqual(result["test"]["users"], 8)
                self.assertTrue(0 <= result["test"]["recall@5"] <= 1)
                self.assertTrue((path.parent / "best_target.pt").exists())
            repeat = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(repeat.returncode, 0, repeat.stdout + repeat.stderr)
            self.assertIn("Reusing teacher", repeat.stdout)
            self.assertEqual(repeat.stdout.count("Already completed"), 4)
            # The dedicated single-file entry point must run B3 by default,
            # without requiring train_mba_kaggle.py as an import dependency.
            dedicated = command[:-5]  # Remove explicit --variants B0 B1 B2 B3.
            dedicated[1] = str(Path(mba.__file__).with_name("train_mba_a_kaggle.py"))
            dedicated[dedicated.index("--output") + 1] = str(root / "mba_a_results")
            improved = subprocess.run(dedicated, capture_output=True, text=True, timeout=90)
            self.assertEqual(improved.returncode, 0, improved.stdout + improved.stderr)
            improved_paths = list((root / "mba_a_results").rglob("result.json"))
            self.assertEqual(len(improved_paths), 1)
            self.assertEqual(json.loads(improved_paths[0].read_text())["variant"], "B3")


if __name__ == "__main__":
    unittest.main()
