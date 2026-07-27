import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from music21 import chord, instrument, note, pitch as m21pitch, stream

from generation import project_paths as _paths
from generation.theme_gen.corpus import (
    default_theme_corpus,
    ingest_dir,
    ingest_file,
    load_corpus,
    music21_corpus,
)
from generation.theme_gen._common import _item_midi, _round_duration
from generation.theme_gen.model import (
    Context,
    MarkovMelodyModel,
    _accent_bucket,
    _phrase_bucket,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_MIDI = ROOT / "assets" / "raw" / "incoming" / "s_first4.mid"


def _music21_part(name, part_instrument, items):
    part = stream.Part(id=name)
    part.partName = name
    part.insert(0, part_instrument)
    for pitches, duration in items:
        event = (
            chord.Chord(pitches, quarterLength=duration)
            if isinstance(pitches, list)
            else note.Note(pitches, quarterLength=duration)
        )
        part.append(event)
    return part


class IngestionTests(unittest.TestCase):
    def test_ingest_file_returns_monophonic_item_list(self):
        items = ingest_file(SAMPLE_MIDI)
        self.assertIsNotNone(items)
        self.assertGreater(sum(float(item[1]) for item in items), 0.0)
        first_sounding = next(item for item in items if item[0] is not None)
        # the pitch name parses under music21 (i.e. it is a valid item pitch)
        self.assertIsInstance(m21pitch.Pitch(first_sounding[0]).midi, int)

    def test_ingest_file_skips_and_warns_on_non_score(self):
        with self.assertWarns(UserWarning):
            self.assertIsNone(ingest_file(ROOT / "README.md"))

    def test_ingest_dir_missing_dir_is_empty(self):
        self.assertEqual(ingest_dir(ROOT / "no" / "such" / "dir"), ())

    def test_picks_lead_over_bass_and_chords(self):
        # Multi-track MIDI like NES VGM: a low bass, a mid chordal pad, and a high lead.
        # The extractor must return the LEAD line, not the bass or the (denser) chords.
        lead_pitches = ["C5", "D5", "E5", "F5", "G5", "A5", "G5", "E5"]
        bass = _music21_part(
            "bass",
            instrument.AcousticBass(),
            [("C2", 1.0), ("E2", 1.0), ("G2", 1.0), ("C2", 1.0)] * 2,
        )
        lead = _music21_part(
            "lead", instrument.Flute(), [(pitch, 1.0) for pitch in lead_pitches]
        )
        chords = _music21_part(
            "chords",
            instrument.Piano(),
            [
                (["C4", "E4", "G4"], 2.0),
                (["F4", "A4", "C5"], 2.0),
                (["G4", "B4", "D5"], 2.0),
                (["C4", "E4", "G4"], 2.0),
            ],
        )
        score = stream.Score((bass, lead, chords))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "multi.mid"
            score.write("midi", fp=str(path))
            line = ingest_file(path)
        self.assertIsNotNone(line)
        names = [name for name, *_ in line if name is not None]
        self.assertEqual(names, lead_pitches)


class LoadCorpusTests(unittest.TestCase):
    def test_themes_source_matches_default_theme_corpus(self):
        self.assertEqual(load_corpus("themes"), tuple(default_theme_corpus()))

    def test_no_sources_defaults_to_themes(self):
        self.assertEqual(load_corpus(), tuple(default_theme_corpus()))

    def test_ingest_source_reads_corpus_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_corpus = Path(tmp)
            shutil.copy(SAMPLE_MIDI, tmp_corpus / "sample.mid")
            with mock.patch.object(_paths, "RAW_CORPUS", tmp_corpus):
                lines = load_corpus("ingest")
        self.assertGreaterEqual(len(lines), 1)

    def test_recursive_category_and_cap(self):
        # a nested category dir (like exotic/<scale>/ or vgm/<game>/) + a per-source cap
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, sub in enumerate(("gameA", "gameB", "gameC")):
                (root / "vgm" / sub).mkdir(parents=True)
                shutil.copy(SAMPLE_MIDI, root / "vgm" / sub / f"t{i}.mid")
            with mock.patch.object(_paths, "RAW_CORPUS", root):
                self.assertEqual(len(load_corpus("vgm")), 3)       # recursive into subfolders
                self.assertEqual(len(load_corpus("vgm@2")), 2)     # capped
                with self.assertRaises(ValueError):
                    load_corpus("nope_not_a_source")

    def test_music21_source_yields_valid_lines(self):
        lines = music21_corpus(limit=3)
        self.assertGreaterEqual(len(lines), 1)
        for line in lines:
            for pitch_name, ql, *_ in line:
                self.assertGreater(float(ql), 0.0)
                if pitch_name is not None:
                    self.assertIsInstance(m21pitch.Pitch(pitch_name).midi, int)

    def test_music21_source_routes_through_load_corpus(self):
        with mock.patch(
            "generation.theme_gen.corpus.music21_corpus",
            return_value=((("C4", 1.0),),),
        ) as patched:
            self.assertEqual(load_corpus("music21"), ((("C4", 1.0),),))
            patched.assert_called_once()


class CliTests(unittest.TestCase):
    def test_build_writes_loadable_artifact(self):
        from generation.tools import build_corpus_model

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "m.json"
            build_corpus_model.main(["--sources", "themes", "--out", str(out)])
            self.assertTrue(out.exists())
            model = MarkovMelodyModel.load(out)
        self.assertGreater(model.interval_weight(0, 2), 0.0)


class ContextFeatureTests(unittest.TestCase):
    def test_accent_bucket(self):
        self.assertEqual(_accent_bucket(0.0, 4.0), "D")
        self.assertEqual(_accent_bucket(2.0, 4.0), "H")
        self.assertEqual(_accent_bucket(1.0, 4.0), "B")
        self.assertEqual(_accent_bucket(1.5, 4.0), "O")

    def test_phrase_bucket(self):
        self.assertEqual(_phrase_bucket(0, 5), "S")
        self.assertEqual(_phrase_bucket(4, 5), "E")
        self.assertEqual(_phrase_bucket(2, 5), "M")

    def test_from_corpus_builds_context_tables_without_changing_first_order(self):
        corpus = default_theme_corpus()
        model = MarkovMelodyModel.from_corpus(corpus)
        self.assertTrue(model.context_interval_counts)
        self.assertTrue(model.context_duration_counts)
        self.assertEqual(model.order, 2)
        self.assertEqual(model.min_count, 20)
        # First-order backbone stays separate from the context tables and still answers.
        self.assertGreater(model.interval_weight(0, 2), 0.0)
        self.assertTrue(all(isinstance(k, int) for k in model.interval_counts))
        self.assertTrue(all(isinstance(k, str) for k in model.context_interval_counts))


class ContextWeightTests(unittest.TestCase):
    # One 4/4 line C4-D4-E4-C4: intervals +2, +2, -4 at offsets 1,2,3.
    LINE = (("C4", 1.0), ("D4", 1.0), ("E4", 1.0), ("C4", 1.0))

    def test_context_weight_conditions_and_differs_from_first_order(self):
        model = MarkovMelodyModel.from_corpus((self.LINE,), min_count=1)
        # Context of the final note (offset 3 -> accent B, phrase E, prev pc E=4, hist 2,2),
        # whose context distribution concentrates on -4, unlike first-order prev_interval=2.
        ctx = Context(history=(2, 2), accent="B", phrase="E", harmonic="4")
        self.assertGreater(
            model.interval_weight(2, -4, context=ctx),
            model.interval_weight(2, -4),
        )

    def test_context_none_matches_first_order(self):
        model = MarkovMelodyModel.from_corpus(default_theme_corpus())
        self.assertEqual(
            model.interval_weight(0, 2, context=None), model.interval_weight(0, 2)
        )
        self.assertEqual(
            model.duration_weight(1.0, 0.5, context=None), model.duration_weight(1.0, 0.5)
        )


class LineScoringTests(unittest.TestCase):
    def test_disable_context_matches_first_order_surprise(self):
        model = MarkovMelodyModel.from_corpus(default_theme_corpus())
        line = default_theme_corpus()[0]
        # Independent first-order surprise sum over the whole line.
        expected = 0.0
        prev_pitch = None
        prev_interval = 0
        prev_duration = None
        for item in line:
            expected += model.transition_surprise(prev_pitch, prev_interval, prev_duration, item)
            midi = _item_midi(item)
            if midi is not None:
                if prev_pitch is not None:
                    prev_interval = midi - prev_pitch
                prev_pitch = midi
            prev_duration = _round_duration(float(item[1]))
        self.assertAlmostEqual(model.line_logprob(line, use_context=False), expected)

    def test_perplexity_is_positive(self):
        model = MarkovMelodyModel.from_corpus(default_theme_corpus())
        self.assertGreater(model.perplexity(default_theme_corpus()), 0.0)


class PerplexityImprovementTests(unittest.TestCase):
    def test_context_model_improves_over_first_order_baseline(self):
        # Default-corpus proxy (D1): Bach chorales; deterministic (sorted corpus paths).
        corpus = list(music21_corpus(corpora=("bach",), limit=12))
        self.assertGreaterEqual(len(corpus), 6)

        # In-sample: the context model strictly improves (richer fit).
        full = MarkovMelodyModel.from_corpus(corpus)
        self.assertLess(
            full.perplexity(corpus, use_context=True),
            full.perplexity(corpus, use_context=False),
        )

        # Held-out: the context model is no worse than the first-order baseline.
        train, held_out = corpus[:-3], corpus[-3:]
        model = MarkovMelodyModel.from_corpus(train)
        self.assertLessEqual(
            model.perplexity(held_out, use_context=True),
            model.perplexity(held_out, use_context=False) + 1e-9,
        )


class ModelSpecTests(unittest.TestCase):
    def test_registry_lookup_and_named_build(self):
        from generation.theme_gen.model_specs import MODEL_SPECS, ModelSpec, get_spec
        from generation.tools import build_corpus_model

        self.assertIn("general_tonal", MODEL_SPECS)
        with self.assertRaises(KeyError):
            get_spec("not_a_spec")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "named.json"
            spec = ModelSpec(("themes",), out=str(out))  # fast source for the test
            path, count = build_corpus_model.build_spec("named", spec)
            self.assertTrue(Path(path).exists())
            self.assertGreater(count, 0)
            MarkovMelodyModel.load(path)  # round-trips


class ModelPersistenceTests(unittest.TestCase):
    def test_round_trip_preserves_context_weights(self):
        line = (("C4", 1.0), ("D4", 1.0), ("E4", 1.0), ("C4", 1.0))
        model = MarkovMelodyModel.from_corpus((line,), min_count=1)
        ctx = Context(history=(2, 2), accent="B", phrase="E", harmonic="4")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ctx.json"
            model.save(path)
            loaded = MarkovMelodyModel.load(path)
        self.assertAlmostEqual(
            loaded.interval_weight(2, -4, context=ctx),
            model.interval_weight(2, -4, context=ctx),
        )

    def test_from_dict_loads_v1_artifact_without_context(self):
        v1 = {
            "version": 1,
            "interval_counts": {"0": {"2": 3}},
            "duration_counts": {"1.0": {"0.5": 2}},
            "global_intervals": {"2": 3},
            "global_durations": {"0.5": 2, "1.0": 1},
        }
        model = MarkovMelodyModel.from_dict(v1)
        self.assertEqual(model.context_interval_counts, {})
        self.assertGreater(model.interval_weight(0, 2), 0.0)

    def test_save_load_round_trip_preserves_weights(self):
        model = MarkovMelodyModel.from_corpus(default_theme_corpus())
        sample = ("C5", 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            self.assertTrue(path.exists())
            loaded = MarkovMelodyModel.load(path)
        self.assertAlmostEqual(loaded.interval_weight(0, 2), model.interval_weight(0, 2))
        self.assertAlmostEqual(loaded.duration_weight(1.0, 0.5), model.duration_weight(1.0, 0.5))
        self.assertAlmostEqual(
            loaded.transition_surprise(72, 0, 1.0, sample),
            model.transition_surprise(72, 0, 1.0, sample),
        )


if __name__ == "__main__":
    unittest.main()
