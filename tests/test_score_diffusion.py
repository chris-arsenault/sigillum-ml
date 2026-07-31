"""Behavioral tests for the phase-10 whole-score self-supervised denoiser.

These are hermetic: roll/dataset/model/baseline/eval behavior is exercised on synthetic score
dicts and a synthetic manifest written to a temp dir, so they do not depend on the gitignored
generated corpus.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from generation.score_diffusion import baselines as B
from generation.score_diffusion import evaluate as E
from generation.score_diffusion import roll as R
from generation.score_diffusion.dataset import (
    WholeScoreRollDataset,
    from_x0,
    iter_batches,
    to_x0,
)
from generation.score_diffusion.model import (
    ScoreDenoiser,
    discretize,
    make_schedule,
    p_losses,
    sample,
)
from generation.score_diffusion import discrete as D


def _score(n_measures=12, parts=None):
    """Synthetic Partitura-shaped score: each measure is 4 quarter-notes long."""
    parts = parts or [
        {"id": "P1", "name": "Violin 1", "abbreviation": "Vln 1", "instruments": [{"name": "Violin"}]},
        {"id": "P2", "name": "Violoncello", "abbreviation": "Vc", "instruments": [{"name": "Cello"}]},
    ]
    measures = [{"index": i + 1, "number": str(i + 1), "offset_ql": f"{i * 4}/1",
                 "duration_ql": "4/1", "implicit": False} for i in range(n_measures)]
    events = []
    for m in range(1, n_measures + 1):
        # Violin: onset on beat 1 (step 0); Cello: onset on beat 3 (ratio .5 -> step 8).
        events.append({"kind": "note", "midi": 76, "measure_index": m,
                       "measure_onset_ql": "0/1", "part_id": "P1"})
        events.append({"kind": "note", "midi": 48, "measure_index": m,
                       "measure_onset_ql": "2/1", "part_id": "P2"})
        events.append({"kind": "rest", "midi": None, "measure_index": m,
                       "measure_onset_ql": "1/1", "part_id": "P1"})
    return {"parts": parts, "measures": measures, "timed_events": events,
            "title": "synthetic", "creators": []}


class RollTests(unittest.TestCase):
    def test_geometry_constants(self):
        self.assertEqual(R.CHANNELS, len(R.FAMILIES) * R.PITCH_COUNT)
        self.assertEqual(R.STEPS, R.WINDOW_MEASURES * R.SUBDIV)
        self.assertEqual(len(R.FAMILIES), 12)

    def test_classify_family_keywords(self):
        self.assertEqual(R.classify_family("Violin 1"), "violin")
        self.assertEqual(R.classify_family("Viola"), "viola")
        self.assertEqual(R.classify_family("Violoncello"), "low_strings")
        self.assertEqual(R.classify_family("Contrabass"), "low_strings")
        self.assertEqual(R.classify_family("Bb Clarinet 1"), "clarinet")
        self.assertEqual(R.classify_family("F Horn 2"), "horn")
        self.assertEqual(R.classify_family("Piccolo"), "flute")
        self.assertEqual(R.classify_family("Timpani"), "percussion")
        # Unknown -> documented fallback.
        self.assertEqual(R.classify_family("Ondes Martenot"), R.FALLBACK_FAMILY)

    def test_meter_normalized_placement(self):
        score = _score()
        roll = R.build_window_roll(score, 1, n_measures=R.WINDOW_MEASURES)
        self.assertEqual(roll.shape, (R.CHANNELS, R.STEPS))
        vln = R.channel_for("violin", 76)
        vc = R.channel_for("low_strings", 48)
        # Violin onset at measure 1 beat 1 -> step 0; cello beat 3 -> step 8.
        self.assertEqual(roll[vln, 0], 1.0)
        self.assertEqual(roll[vc, 8], 1.0)
        # Measure 2 violin onset -> step SUBDIV.
        self.assertEqual(roll[vln, R.SUBDIV], 1.0)

    def test_channel_range_guard(self):
        self.assertIsNone(R.channel_for("violin", R.LOW_MIDI - 1))
        self.assertIsNone(R.channel_for("violin", R.HIGH_MIDI + 1))
        self.assertEqual(R.channel_for("flute", R.LOW_MIDI), 0)

    def test_roundtrip_events(self):
        score = _score()
        roll = R.build_window_roll(score, 1)
        events = R.roll_to_events(roll, start_measure=1)
        rebuilt = np.zeros_like(roll)
        for e in events:
            ch = R.channel_for(e["family"], e["midi"])
            step = (e["measure_index"] - 1) * R.SUBDIV + e["step_in_measure"]
            rebuilt[ch, step] = 1.0
        self.assertTrue(np.array_equal(roll > 0, rebuilt > 0))

    def test_family_time_activity(self):
        score = _score()
        roll = R.build_window_roll(score, 1)
        fta = R.family_time_activity(roll)
        self.assertEqual(fta.shape, (len(R.FAMILIES), R.STEPS))
        self.assertEqual(int(fta.sum()), int((roll > 0).sum()))


def _write_corpus(tmp: Path):
    """Write a synthetic manifest + observations with two lineages per split."""
    records = []
    base = tmp
    (base / "observations").mkdir(parents=True, exist_ok=True)
    plan = [("train", "lin_a", 40), ("train", "lin_b", 30),
            ("validation", "lin_c", 24), ("test", "lin_d", 20)]
    for i, (split, lineage, nmeas) in enumerate(plan):
        obs = {"score": _score(n_measures=nmeas)}
        rel = f"observations/score_{i}.json"
        (base / rel).write_text(json.dumps(obs))
        records.append({"observation_file": rel, "split": split, "lineage_id": lineage,
                        "source_id": f"src_{i}", "score_path": f"x/{i}.mxl"})
    manifest = {"records": records, "manifest_digest": "sha256:test", "schema_version": 1}
    (base / "manifest.json").write_text(json.dumps(manifest))
    return base / "manifest.json"


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.manifest = _write_corpus(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_split_isolation_and_shapes(self):
        train = WholeScoreRollDataset(self.manifest, "train", min_active=1)
        val = WholeScoreRollDataset(self.manifest, "validation", min_active=1)
        test = WholeScoreRollDataset(self.manifest, "test", min_active=1)
        self.assertEqual(train.coverage()["records"], 2)
        self.assertEqual(val.coverage()["records"], 1)
        self.assertEqual(test.coverage()["records"], 1)
        self.assertEqual(train.coverage()["lineages"], 2)
        self.assertGreater(len(train), 0)
        x = train[0]
        self.assertEqual(tuple(x.shape), (R.CHANNELS, R.STEPS))
        self.assertGreaterEqual(float(x.min()), -1.0)
        self.assertLessEqual(float(x.max()), 1.0)

    def test_x0_roundtrip_and_batches(self):
        ds = WholeScoreRollDataset(self.manifest, "train", min_active=1)
        r = ds.roll01(0)
        self.assertTrue(np.array_equal(from_x0(to_x0(r)), r))
        batch = next(iter_batches(ds, 4, shuffle=False))
        self.assertEqual(batch.shape[0], min(4, len(ds)))
        self.assertEqual(tuple(batch.shape[1:]), (R.CHANNELS, R.STEPS))

    def test_channel_marginals(self):
        ds = WholeScoreRollDataset(self.manifest, "train", min_active=1)
        marg = ds.channel_marginals()
        self.assertEqual(marg.shape, (R.CHANNELS,))
        self.assertTrue((marg >= 0).all() and (marg <= 1).all())
        # The two active channels (violin 76, cello 48) must be the most frequent.
        vln = R.channel_for("violin", 76)
        self.assertGreater(marg[vln], 0.0)

    def test_min_active_filter(self):
        strict = WholeScoreRollDataset(self.manifest, "train", min_active=10_000)
        self.assertEqual(len(strict), 0)


class ModelTests(unittest.TestCase):
    def test_forward_and_loss(self):
        model = ScoreDenoiser(width=32, depth=2)
        sched = make_schedule(steps=10)
        x0 = torch.zeros(2, R.CHANNELS, R.STEPS)
        x0[:, :20, ::8] = 1.0
        x0 = x0 * 2 - 1
        t = torch.randint(0, 10, (2,))
        out = model(x0, t)
        self.assertEqual(tuple(out.shape), (2, R.CHANNELS, R.STEPS))
        loss = p_losses(model, x0, sched)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_sample_shape_and_discretize(self):
        model = ScoreDenoiser(width=32, depth=2)
        sched = make_schedule(steps=8)
        s = sample(model, sched, n=1, device="cpu")
        self.assertEqual(tuple(s.shape), (1, R.CHANNELS, R.STEPS))
        d = discretize(s)
        self.assertTrue(set(torch.unique(d).tolist()).issubset({0.0, 1.0}))

    def test_anchor_preserved(self):
        model = ScoreDenoiser(width=32, depth=2)
        sched = make_schedule(steps=8)
        x0 = torch.zeros(1, R.CHANNELS, R.STEPS)
        x0[:, :10, ::4] = 1.0
        x0 = x0 * 2 - 1
        mask = torch.zeros(1, R.CHANNELS, R.STEPS)
        mask[:, :, : R.STEPS // 2] = 1.0
        out = sample(model, sched, n=1, anchor=(mask, x0), device="cpu")
        self.assertTrue(torch.allclose(out[mask.bool()], x0[mask.bool()]))


class BaselineTests(unittest.TestCase):
    def test_silence(self):
        ctx = np.ones((R.CHANNELS, R.STEPS), dtype=np.float32)
        mask = np.ones_like(ctx)
        self.assertEqual(B.SilenceBaseline().predict(ctx, mask).sum(), 0.0)

    def test_marginal_calibration_density(self):
        marg = np.zeros(R.CHANNELS)
        marg[:100] = np.linspace(0.5, 0.01, 100)
        target = 10 / R.CHANNELS
        bl = B.MarginalBaseline.calibrate(marg, target)
        pred = bl.predict(np.zeros((R.CHANNELS, R.STEPS), dtype=np.float32),
                          np.ones((R.CHANNELS, R.STEPS), dtype=np.float32))
        active_channels = int((pred[:, 0] > 0.5).sum())
        self.assertAlmostEqual(active_channels, 10, delta=1)

    def test_persistence_copies_previous_block(self):
        ctx = np.zeros((R.CHANNELS, R.STEPS), dtype=np.float32)
        # A block of activity in the first two measures.
        ctx[5, : R.SUBDIV] = 1.0
        mask = np.zeros_like(ctx)
        mask[:, R.SUBDIV:2 * R.SUBDIV] = 1.0   # mask the second measure
        pred = B.PersistenceBaseline(period=R.SUBDIV).predict(ctx, mask)
        # The masked second measure should copy the first measure's activity.
        self.assertTrue(np.array_equal(pred[5, R.SUBDIV:2 * R.SUBDIV], ctx[5, :R.SUBDIV]))


class EvaluateTests(unittest.TestCase):
    def test_block_mask(self):
        mask = E.make_block_mask(mask_measures=2)
        self.assertEqual(mask.shape, (R.CHANNELS, R.STEPS))
        self.assertEqual(int(mask.sum()), R.CHANNELS * 2 * R.SUBDIV)

    def test_prf_perfect_and_empty(self):
        truth = np.zeros((R.CHANNELS, R.STEPS), dtype=np.float32)
        truth[3, 5] = 1.0
        truth[7, 5] = 1.0
        mask = np.ones_like(truth)
        perfect = E.active_cell_prf(truth.copy(), truth, mask)
        self.assertEqual(perfect["f1"], 1.0)
        empty = E.active_cell_prf(np.zeros_like(truth), truth, mask)
        self.assertEqual(empty["f1"], 0.0)
        self.assertEqual(empty["recall"], 0.0)

    def test_repetition_rate(self):
        constant = np.zeros((R.CHANNELS, R.STEPS), dtype=np.float32)
        constant[10, :] = 1.0            # same single-onset column everywhere
        self.assertGreater(E.repetition_rate(constant), 0.9)
        varied = np.zeros((R.CHANNELS, R.STEPS), dtype=np.float32)
        for s in range(R.STEPS):
            varied[s % 50, s] = 1.0      # a different column each step
        self.assertLess(E.repetition_rate(varied), 0.1)

    def test_anti_collapse_flags_noise(self):
        # A "sampler" that returns pure noise (half cells active) must flag repetition/silence sanely.
        tmp = tempfile.TemporaryDirectory()
        manifest = _write_corpus(Path(tmp.name))
        ds = WholeScoreRollDataset(manifest, "validation", min_active=1)
        rng = np.random.default_rng(0)
        report = E.anti_collapse_report(
            ds, lambda: (rng.random((R.CHANNELS, R.STEPS)) > 0.5).astype(np.float32),
            n_samples=3, limit_authentic=5)
        self.assertGreater(report["generated"]["density_mean"], report["authentic"]["density_mean"])
        tmp.cleanup()


class DiscreteDiffusionTests(unittest.TestCase):
    def test_survival_schedule_endpoints(self):
        sched = D.make_survival_schedule(steps=50)
        self.assertEqual(sched["steps"], 50)
        self.assertGreater(float(sched["abar"][0]), 0.9)     # near clean at t=0
        self.assertLess(float(sched["abar"][-1]), 0.05)      # near silence at t=T-1

    def test_q_sample_is_subset_of_x0(self):
        sched = D.make_survival_schedule(steps=20)
        x0 = torch.zeros(3, R.CHANNELS, R.STEPS)
        x0[:, ::7, ::5] = 1.0
        t = torch.tensor([0, 10, 19])
        xt = D.q_sample(x0, t, sched, generator=torch.Generator().manual_seed(0))
        # Corruption only removes onsets, never invents them.
        self.assertTrue(bool(((xt > 0) & (x0 < 0.5)).sum() == 0))
        # Later timesteps keep fewer onsets than earlier ones (in expectation; here strong).
        self.assertGreaterEqual(float(xt[0].sum()), float(xt[2].sum()))

    def test_loss_finite_and_backward(self):
        model = D.OccupancyDenoiser(width=32, depth=2)
        sched = D.make_survival_schedule(steps=10)
        x0 = torch.zeros(2, R.CHANNELS, R.STEPS)
        x0[:, :30, ::4] = 1.0
        loss = D.p_losses(model, x0, sched, pos_weight=20.0,
                          generator=torch.Generator().manual_seed(1))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_sample_binary_and_shape(self):
        model = D.OccupancyDenoiser(width=32, depth=2)
        sched = D.make_survival_schedule(steps=8)
        out = D.sample(model, sched, n=1, device="cpu",
                       generator=torch.Generator().manual_seed(2))
        self.assertEqual(tuple(out.shape), (1, R.CHANNELS, R.STEPS))
        self.assertTrue(set(torch.unique(out).tolist()).issubset({0.0, 1.0}))

    def test_anchor_context_preserved(self):
        model = D.OccupancyDenoiser(width=32, depth=2)
        sched = D.make_survival_schedule(steps=8)
        ctx = torch.zeros(1, R.CHANNELS, R.STEPS)
        ctx[:, 5, 3] = 1.0
        ctx[:, 40, 20] = 1.0
        out = D.sample(model, sched, n=1, anchor=(ctx,), device="cpu",
                       generator=torch.Generator().manual_seed(3))
        # Every anchored context onset must remain active.
        self.assertTrue(bool((out[ctx > 0] > 0.5).all()))

    def test_reconstructor_returns_binary_full_grid(self):
        model = D.OccupancyDenoiser(width=32, depth=2)
        sched = D.make_survival_schedule(steps=6)
        recon = D.discrete_reconstructor(model, sched)
        context = np.zeros((R.CHANNELS, R.STEPS), dtype=np.float32)
        context[5, :R.SUBDIV] = 1.0
        mask = E.make_block_mask(mask_measures=2)
        pred = recon(context, mask)
        self.assertEqual(pred.shape, (R.CHANNELS, R.STEPS))
        self.assertTrue(set(np.unique(pred).tolist()).issubset({0.0, 1.0}))


if __name__ == "__main__":
    unittest.main()
