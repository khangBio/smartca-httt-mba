"""Correctness checks for the research implementation; standard-library unittest."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_mba as mba
from summarize_results import summarize


class DataTests(unittest.TestCase):
    def test_split_reproducible_disjoint_preserves_training_user(self):
        pairs = np.array([[u, i] for u in range(4) for i in range(u + 1)])
        fit, val = mba.split_by_user(pairs, 0.2, 7)
        again_fit, again_val = mba.split_by_user(pairs, 0.2, 7)
        np.testing.assert_array_equal(fit, again_fit)
        np.testing.assert_array_equal(val, again_val)
        self.assertEqual(set(fit[:, 0]), set(pairs[:, 0]))
        self.assertFalse(np.isin(mba.keys(fit, 10), mba.keys(val, 10)).any())
        self.assertEqual(len(fit) + len(val), len(pairs))

    def test_negative_samples_never_include_training_positives(self):
        observed = np.array([[0, 0], [0, 1], [0, 2], [1, 2], [1, 3]])
        sampler = mba.NegativeSampler(observed, 2, 4, np.random.default_rng(4), 0.7)
        users = np.array([0, 1] * 300)
        negatives = sampler.sample(users, 4)
        query = np.repeat(users, 4) * 4 + negatives.ravel()
        self.assertFalse(mba.contains(np.sort(mba.keys(observed, 4)), query).any())
        self.assertTrue((negatives[users == 0] == 3).all())
        full = mba.NegativeSampler(np.array([[0, 0], [0, 1]]), 1, 2, np.random.default_rng(1))
        with self.assertRaises(ValueError):
            full.sample([0])

    def test_input_reindexing_no_target_or_click_validation_in_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = dict(buy_train=[[10, 100], [10, 300], [20, 200], [20, 400]],
                       buy_test=[[10, 900], [20, 800]],
                       pv_train=[[10, i] for i in (100, 300, 500, 600)] +
                                [[20, i] for i in (200, 400, 700, 800)])
            paths = {}
            for name, rows in raw.items():
                paths[name] = root / f"taobao_{name}.csv"
                pd.DataFrame(rows, columns=["uid", "sid"]).to_csv(paths[name], sep="\t", index=False)
            data = mba.prepare_data(paths, 0.3, 2026)
            self.assertEqual(data["n_users"], 2)
            self.assertEqual(data["n_items"], 9)
            self.assertFalse(mba.contains(np.sort(mba.keys(data["buy_fit"], 9)), mba.keys(data["buy_val"], 9)).any())
            self.assertFalse(mba.contains(np.sort(mba.keys(data["pv_fit"], 9)), mba.keys(data["pv_val"], 9)).any())
            # No automatic union of held-out purchase labels into the click graph.
            only_test_item = np.searchsorted(data["item_ids"], 900)
            self.assertNotIn(only_test_item, data["pv_fit"][:, 1])
            self.assertEqual(set(mba.discover_data(root, "taobao")), set(paths))


class ObjectiveTests(unittest.TestCase):
    def test_kl_matches_probability_formula_and_stable_extremes(self):
        t = torch.tensor([-2.0, 0.3, 4.0])
        s = torch.tensor([0.1, -1.0, 2.0], requires_grad=True)
        p, q = t.sigmoid(), s.sigmoid()
        expected = p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())
        torch.testing.assert_close(mba.bernoulli_kl(t, s), expected)
        extreme = mba.bernoulli_kl(torch.tensor([-1000., 1000.]), torch.tensor([1000., -1000.]))
        self.assertTrue(torch.isfinite(extreme).all())
        self.assertTrue((extreme >= 0).all())

    def test_denoising_formula_and_phase_gradients(self):
        for phase in ["DP", "DN"]:
            target = torch.tensor([0.1, -0.2], requires_grad=True)
            h1 = torch.tensor([0.4, -0.6], requires_grad=True)
            h2 = torch.tensor([0.5, -0.7], requires_grad=True)
            labels = torch.tensor([1, 0])
            result = mba.denoising_nll(target, h1, h2, labels, phase, 3, 5)
            p = target.sigmoid()
            expected = (-h1.sigmoid().log() * p + 3 * (1 - p)) if phase == "DN" else (-h2.sigmoid().log() * (1 - p))
            expected_neg = (-(-h1).sigmoid().log() * p) if phase == "DN" else (-(-h2).sigmoid().log() * (1 - p) + 5 * p)
            torch.testing.assert_close(result, torch.stack([expected[0], expected_neg[1]]))
            result.sum().backward()
            self.assertIsNotNone(target.grad)
            self.assertEqual(h1.grad is None, phase == "DP")
            self.assertEqual(h2.grad is None, phase == "DN")

    def test_graph_is_train_only_normalized_and_differentiable(self):
        pairs = np.array([[0, 0], [0, 1], [1, 1]])
        graph = mba.make_graph(pairs, 2, 3, "cpu")
        dense = graph.to_dense()
        torch.testing.assert_close(dense, dense.T)
        self.assertEqual(int((dense != 0).sum()), 6)
        self.assertAlmostEqual(float(dense[0, 2]), 1 / np.sqrt(2), places=6)
        model = mba.Recommender(2, 3, 4, 2, graph)
        u, i = model.encode()
        (u.sum() + i.sum()).backward()
        self.assertIsNotNone(model.user.weight.grad)
        self.assertNotIn("graph", model.state_dict())


class RankingTests(unittest.TestCase):
    def test_metrics_by_hand(self):
        prediction = np.array([[4, 3, 2], [1, 4, 2]])
        result = mba.metrics_at_k(prediction, [[3, 4], [2]], [1, 3])
        np.testing.assert_allclose(result["Recall@1"], [0.5, 0.0])
        np.testing.assert_allclose(result["Recall@3"], [1.0, 1.0])
        np.testing.assert_allclose(result["NDCG@3"], [1.0, 0.5])

    def test_chunked_topk_matches_dense_and_masks_training_only(self):
        model = mba.Recommender(2, 8, 2)
        teacher = mba.Recommender(2, 8, 2)
        with torch.no_grad():
            model.user.weight.copy_(torch.tensor([[1., .3], [-.4, 1.]]))
            model.item.weight.copy_(torch.tensor([[1., .1], [.3, .8], [-1., 2.], [.7, -.5],
                                                 [.2, -1.], [-.5, -.6], [1.2, 1.4], [-.3, 1.5]]))
            teacher.load_state_dict(model.state_dict())
        truth = np.array([[0, 3], [1, 7]])
        seen = np.array([[0, 6], [1, 2]])
        a, users_a, per_a = mba.evaluate(model, truth, seen, [1, 3], 1, 2, teacher, .8)
        b, users_b, per_b = mba.evaluate(model, truth, seen, [1, 3], 2, 8, teacher, .8)
        self.assertEqual(a, b)
        np.testing.assert_array_equal(users_a, users_b)
        for name in per_a:
            np.testing.assert_array_equal(per_a[name], per_b[name])
        with torch.no_grad():
            score = (model.user.weight @ model.item.weight.T).sigmoid()
            score[0, 6] = -float("inf")
            score[1, 2] = -float("inf")
            expected = mba.metrics_at_k(score.topk(3).indices.numpy(), [[3], [7]], [1, 3])
        for name in per_a:
            np.testing.assert_allclose(per_a[name], expected[name])


class EndToEndTests(unittest.TestCase):
    def test_all_stages_mf_and_lightgcn(self):
        # Includes clicked-but-not-bought batches, both DP/DN phases and inference reload.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "data"
            source.mkdir()
            tables = dict(buy_train=[[u, (u * 7 + j) % 40] for u in range(6) for j in range(4)],
                          buy_test=[[u, (u * 7 + 5) % 40] for u in range(6)],
                          pv_train=[[u, (u * 7 + j) % 40] for u in range(6) for j in range(12)])
            for name, pairs in tables.items():
                pd.DataFrame(pairs, columns=["uid", "sid"]).to_csv(source / f"taobao_{name}.csv", sep="\t", index=False)
            for variant, backbone in [("mba", "mf"), ("adaptive", "mf"), ("adaptive", "lightgcn")]:
                config = mba.parser().parse_args(["--data-root", str(source), "--output", str(root / "output"),
                    "--teacher-cache", str(root / "cache"),
                    "--variant", variant, "--backbone", backbone, "--device", "cpu", "--cpu-threads", "1",
                    "--epochs", "2", "--pretrain-epochs", "2", "--batch-size", "8", "--dim", "4",
                    "--top-k", "2", "4", "--select-metric", "Recall@4", "--item-chunk", "7", "--eval-every", "1"])
                result = mba.run(config)
                self.assertFalse(result["debug"])
                self.assertTrue(all(np.isfinite(v) for v in result["test"].values()))
                folder = root / "output" / "taobao" / f"{variant}_{backbone}_seed2020"
                history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
                self.assertEqual(len(history), 4 if variant == "adaptive" and backbone == "mf" else 6)
                if variant == "adaptive" and backbone == "mf":
                    self.assertIn("reused_cache", history[0])
                self.assertTrue((folder / "target_best.pt").is_file())
                self.assertEqual(len(np.load(folder / "test_per_user.npz")["users"]), 6)
            debug = root / "output" / "debug"
            debug.mkdir()
            (debug / "result.json").write_text(json.dumps(dict(result, debug=True)))
            summarize(root / "output", resamples=32)
            summary = pd.read_csv(root / "output" / "all_runs.csv")
            self.assertEqual(len(summary), 3)
            comparison = pd.read_csv(root / "output" / "paired_differences.csv")
            self.assertEqual(len(comparison), 4)
            self.assertTrue((comparison["paired_seeds"] == 1).all())
            self.assertTrue((comparison["user_bootstrap_ci_low"] <= comparison["user_bootstrap_ci_high"]).all())


if __name__ == "__main__":
    unittest.main()
