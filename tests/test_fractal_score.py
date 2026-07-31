import unittest

import numpy as np
import torch

from generation.fractal_score import (
    Bigram,
    BigramLM,
    CopyNearestPillar,
    HarmonyProgression,
    HarmonyVocab,
    RefinementConfig,
    RefinementOperator,
    RefinementSchedule,
    TrainConfig,
    Unigram,
    evaluate_recursive,
    evaluate_steps,
    extract_windows,
    recursive_refine,
    refinement_arrays,
    split_windows,
    train_operator,
)
from generation.fractal_score.dataset import movement_holdout
from generation.fractal_score.harmony import NONE_TOKEN, normalize_roman
from generation.fractal_score.ladder import (
    LadderError,
    refinement_positions,
    revealed_positions,
)
from generation.fractal_score.vocab import RARE_TOKEN


def _progression(score_id, lineage, split, tokens):
    return HarmonyProgression(
        score_id=score_id,
        lineage_id=lineage,
        split=split,
        home_key="C",
        first_measure=1,
        tokens=tuple(tokens),
    )


def _corpus():
    pattern = ["I", "V", "vi", "IV", "ii", "V", "I", "V/V"]
    progressions = []
    for lineage in ("mozart", "haydn", "dvorak"):
        for movement in range(3):
            tokens = (pattern * 12)[: 60 + movement * 4]
            progressions.append(
                _progression(f"{lineage}_{movement}", lineage, "train", tokens)
            )
    return progressions


class LadderTests(unittest.TestCase):
    def test_geometric_schedule(self):
        schedule = RefinementSchedule.geometric(8)
        self.assertEqual((8, 4, 2, 1), schedule.strides)
        self.assertEqual(((8, 4), (4, 2), (2, 1)), schedule.steps())
        self.assertEqual(0, schedule.level_index(8))
        self.assertEqual(3, schedule.level_index(1))

    def test_schedule_validation(self):
        with self.assertRaises(LadderError):
            RefinementSchedule((8, 3, 1))
        with self.assertRaises(LadderError):
            RefinementSchedule((8, 4, 2))

    def test_refinement_positions_are_new_child_slots(self):
        self.assertEqual([0, 4, 8], revealed_positions(12, 4))
        self.assertEqual([2, 6, 10], refinement_positions(12, 4, 2))
        self.assertEqual(
            sorted(revealed_positions(12, 2)),
            sorted(revealed_positions(12, 4) + refinement_positions(12, 4, 2)),
        )


class HarmonyRepresentationTests(unittest.TestCase):
    def test_normalize_roman_collapses_figures(self):
        self.assertEqual("I", normalize_roman("I64"))
        self.assertEqual("V", normalize_roman("V7"))
        self.assertEqual("ii", normalize_roman("ii65"))
        self.assertEqual("V/V", normalize_roman("V7/V"))
        self.assertEqual("viio", normalize_roman("viio7"))

    def test_vocab_is_key_relative_and_handles_rare(self):
        vocab = HarmonyVocab.build(_corpus(), max_tokens=12)
        for token in vocab.tokens:
            self.assertNotIn(":", token)
        self.assertEqual(vocab.rare_id, vocab.encode("never-seen-chord"))
        self.assertEqual(RARE_TOKEN, vocab.decode(vocab.rare_id))
        self.assertEqual("I", vocab.decode(vocab.encode("I")))


class DatasetTests(unittest.TestCase):
    def test_windows_and_short_padding(self):
        vocab = HarmonyVocab.build(_corpus(), max_tokens=40)
        short = [_progression("s", "l", "train", ["I", "V", "I"])]
        windows = extract_windows(short, vocab, window=8, stride=4)
        self.assertEqual(1, len(windows))
        self.assertEqual(8, len(windows[0]))
        self.assertEqual(vocab.pad_id, int(windows[0].token_ids[-1]))

    def test_refinement_arrays_reveal_parent_mask_child(self):
        vocab = HarmonyVocab.build(_corpus(), max_tokens=40)
        tokens = np.array([vocab.encode("I")] * 8, dtype=np.int64)
        inputs, target, loss_mask = refinement_arrays(
            tokens, parent_stride=4, child_stride=2, mask_id=vocab.mask_id, pad_id=vocab.pad_id
        )
        self.assertEqual(vocab.encode("I"), int(inputs[0]))
        self.assertEqual(vocab.encode("I"), int(inputs[4]))
        self.assertEqual(vocab.mask_id, int(inputs[2]))
        self.assertTrue(bool(loss_mask[2]) and bool(loss_mask[6]))
        self.assertFalse(bool(loss_mask[0]) or bool(loss_mask[4]))
        np.testing.assert_array_equal(target, tokens)

    def test_movement_holdout_puts_every_lineage_in_train(self):
        split = movement_holdout(_corpus(), seed=1)
        train_lineages = {p.lineage_id for p in split if p.split == "train"}
        self.assertEqual({"mozart", "haydn", "dvorak"}, train_lineages)
        self.assertTrue(any(p.split == "test" for p in split))
        self.assertTrue(any(p.split == "validation" for p in split))


class BaselineTests(unittest.TestCase):
    def test_copy_nearest_pillar_copies_left(self):
        vocab = HarmonyVocab.build(_corpus(), max_tokens=40)
        tokens = np.array(
            [vocab.encode(t) for t in ["I", "x", "V", "x", "vi", "x", "IV", "x"]],
            dtype=np.int64,
        )
        filled = CopyNearestPillar().fill(tokens, parent_stride=2, child_stride=1)
        self.assertEqual(int(tokens[0]), int(filled[1]))
        self.assertEqual(int(tokens[2]), int(filled[3]))

    def test_ngram_baselines_fit(self):
        vocab = HarmonyVocab.build(_corpus(), max_tokens=40)
        windows = extract_windows(_corpus(), vocab, window=16, stride=8)
        ignore = [vocab.pad_id, vocab.mask_id]
        self.assertIsInstance(Unigram.fit(windows, ignore).most_common_id, int)
        self.assertIn(vocab.encode("V"), Bigram.fit(windows, ignore).transitions.values())
        lm = BigramLM.fit(windows, len(vocab), ignore)
        self.assertLess(lm.log_prob(vocab.encode("I"), vocab.encode("V")), 0.0)


class OperatorTests(unittest.TestCase):
    def test_train_evaluate_and_preserve_pillars(self):
        torch.manual_seed(0)
        corpus = _corpus()
        vocab = HarmonyVocab.build(corpus, max_tokens=40)
        windows = extract_windows(corpus, vocab, window=16, stride=8)
        grouped = split_windows(windows)
        schedule = RefinementSchedule.geometric(8)
        config = RefinementConfig(
            vocab_size=len(vocab),
            key_count=vocab.key_count,
            level_count=schedule.level_count,
            max_length=16,
            d_model=32,
            layers=2,
            heads=2,
            ff=64,
        )
        model, history = train_operator(
            grouped["train"], schedule, vocab, config, TrainConfig(epochs=2, batch_size=16)
        )
        self.assertEqual(2, len(history))
        baselines = [CopyNearestPillar()]
        steps = evaluate_steps(model, vocab, grouped["train"], schedule, baselines)
        self.assertEqual(len(schedule.steps()), len(steps))
        for step in steps:
            self.assertGreaterEqual(step.learned_accuracy, 0.0)
            self.assertLessEqual(step.learned_accuracy, 1.0)
        recursive = evaluate_recursive(model, vocab, grouped["train"], schedule, baselines)
        self.assertTrue(recursive.parent_preserved)

    def test_recursive_refine_holds_pillars_fixed(self):
        torch.manual_seed(0)
        corpus = _corpus()
        vocab = HarmonyVocab.build(corpus, max_tokens=40)
        schedule = RefinementSchedule.geometric(8)
        config = RefinementConfig(
            vocab_size=len(vocab),
            key_count=vocab.key_count,
            level_count=schedule.level_count,
            max_length=32,
            d_model=32,
            layers=1,
            heads=2,
            ff=64,
        )
        model = RefinementOperator(config)
        truth = np.array(
            [vocab.encode(t) for t in (["I", "V", "vi", "IV"] * 8)], dtype=np.int64
        )
        generated = recursive_refine(model, vocab, truth, vocab.encode_key("C"), schedule)
        for position in revealed_positions(truth.shape[0], schedule.coarsest):
            self.assertEqual(int(truth[position]), int(generated[position]))
        self.assertEqual(truth.shape, generated.shape)


if __name__ == "__main__":
    unittest.main()
