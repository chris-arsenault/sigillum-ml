import tempfile
import unittest
from pathlib import Path

from generation.theme_gen import frame, generate_theme_batch, kernel, pin
from generation.tools import gen_themes

TINY_KERNEL_SRC = (
    "from generation.theme_gen import frame, pin, kernel\n"
    "KERNEL = kernel(frame(2, lower='C4', upper='C6', durations=(0.5, 1.0, 1.5, 2.0)),\n"
    "                pins=[pin(1, 1.0, ('C4', 1.0), ('E4', 1.0), ('G4', 1.0), ('C5', 1.0))])\n"
)


def tiny_kernel():
    return kernel(
        frame(2, lower="C4", upper="C6", durations=(0.5, 1.0, 1.5, 2.0)),
        pins=[pin(1, 1.0, ("C4", 1.0), ("E4", 1.0), ("G4", 1.0), ("C5", 1.0))],
    )


class GenThemesTests(unittest.TestCase):
    def test_load_kernel_by_dotted_path(self):
        from tests import sample_kernel

        self.assertEqual(gen_themes.load_kernel("tests.sample_kernel"), sample_kernel.KERNEL)

    def test_load_kernel_by_file_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "k.py"
            path.write_text(TINY_KERNEL_SRC)
            loaded = gen_themes.load_kernel(str(path))
            self.assertEqual(loaded.frame.bars, 2)

    def test_generate_matches_direct(self):
        k = tiny_kernel()
        cli = gen_themes.generate(k, seed=5, pool=20, batch=4)
        direct = generate_theme_batch(k, pool_size=20, batch_size=4, seed=5)
        self.assertEqual([c.items for c in cli], [c.items for c in direct])

    def test_main_writes_score_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            kernel_file = Path(tmp) / "k.py"
            kernel_file.write_text(TINY_KERNEL_SRC)
            out = Path(tmp) / "out"
            gen_themes.main([str(kernel_file), "--out", str(out), "--pool", "20", "--batch", "4", "--seed", "5"])
            self.assertTrue(list(out.glob("*.mid")))
            self.assertTrue(list(out.glob("*.musicxml")))
            self.assertTrue(list(out.glob("*.md")))

    def test_load_model_and_generate_with_it(self):
        from generation.theme_gen.corpus import default_theme_corpus
        from generation.theme_gen.model import MarkovMelodyModel

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            MarkovMelodyModel.from_corpus(default_theme_corpus()).save(path)
            loaded = gen_themes.load_model(str(path))
        self.assertIsNotNone(loaded)
        batch = gen_themes.generate(tiny_kernel(), seed=5, pool=20, batch=4, model=loaded)
        self.assertEqual(len(batch), 4)

    def test_missing_model_falls_back(self):
        self.assertIsNone(gen_themes.load_model("a_model_that_was_never_built"))

    def test_model_choice_changes_generation(self):
        # Two different models on the same kernel+seed yield different candidates — proof the
        # chosen model actually drives generation (not just the pins).
        from generation.theme_gen.model import MarkovMelodyModel

        stepwise = MarkovMelodyModel.from_corpus([[("C4", 0.5), ("D4", 0.5), ("E4", 0.5), ("F4", 0.5)] * 6])
        leapy = MarkovMelodyModel.from_corpus([[("C4", 0.5), ("C5", 0.5), ("C4", 0.5), ("G5", 0.5)] * 6])
        kernel = tiny_kernel()
        a = gen_themes.generate(kernel, seed=5, pool=20, batch=4, model=stepwise)
        b = gen_themes.generate(kernel, seed=5, pool=20, batch=4, model=leapy)
        self.assertNotEqual([c.items for c in a], [c.items for c in b])

    def test_cli_regenerates_from_dsl_kernel(self):
        # EXIT: the CLI loads a DSL kernel by dotted path and regenerates a valid batch with no
        # edits to generator/library code. Small pool for speed; identity rests on kernel-equality
        # + seed determinism.
        dsl_kernel = gen_themes.load_kernel("tests.sample_kernel")
        batch = gen_themes.generate(dsl_kernel, seed=20260613, pool=16, batch=3)
        self.assertEqual(len(batch), 3)
        # Hard pitch/rhythm pins hold on every candidate; harmonic pins are soft (guided).
        from generation.theme_gen.engine import pin_outcome
        self.assertTrue(all(pin_outcome(dsl_kernel, c.items)[0] for c in batch))


if __name__ == "__main__":
    unittest.main()
