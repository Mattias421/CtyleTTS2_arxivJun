import tempfile
import unittest
from pathlib import Path

from scripts.create_esd_splits import (
    EMOTIONS,
    ENGLISH_SPEAKERS,
    TARGET_MINUTES,
    Utterance,
    build_splits,
    closest_duration_prefix,
    epoch_schedule,
    validate_splits,
    write_manifest,
)


def make_utterances(duration=3.0):
    records = []
    for sentence_id in range(1, 351):
        for speaker_id, speaker in enumerate(ENGLISH_SPEAKERS):
            for emotion in EMOTIONS:
                utterance_id = sentence_id + EMOTIONS.index(emotion) * 350
                records.append(
                    Utterance(
                        wav_path=(
                            f"ESD/{speaker}/{emotion}/"
                            f"{speaker}_{utterance_id:06d}.wav"
                        ),
                        source_wav_path=Path("unused.wav"),
                        text=f"sentence {sentence_id}",
                        speaker=speaker,
                        speaker_id=speaker_id,
                        emotion=emotion,
                        base_sentence_id=sentence_id,
                        duration_seconds=duration,
                    )
                )
    return records


class EsdSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.utterances = make_utterances()

    def test_splits_are_deterministic_nested_balanced_and_disjoint(self):
        first, first_ids = build_splits(self.utterances, seed=1234)
        second, second_ids = build_splits(self.utterances, seed=1234)

        self.assertEqual(first_ids, second_ids)
        self.assertEqual(
            [record.wav_path for record in first["train_60m"]],
            [record.wav_path for record in second["train_60m"]],
        )
        validate_splits(first)

        previous = set()
        for minutes in TARGET_MINUTES:
            records = first[f"train_{minutes}m"]
            paths = {record.wav_path for record in records}
            self.assertTrue(previous.issubset(paths))
            self.assertLessEqual(
                abs(sum(record.duration_seconds for record in records) - minutes * 60),
                1.5,
            )
            previous = paths

        for split_name in ("val", "test"):
            self.assertEqual(len(first[split_name]), 100)

    def test_different_seed_changes_selection(self):
        first, first_ids = build_splits(self.utterances, seed=1)
        second, second_ids = build_splits(self.utterances, seed=2)
        self.assertNotEqual(first_ids, second_ids)
        self.assertNotEqual(
            [record.wav_path for record in first["train_1m"]],
            [record.wav_path for record in second["train_1m"]],
        )

    def test_closest_prefix_prefers_smaller_prefix_on_tie(self):
        records = make_utterances(duration=4.0)[:3]
        self.assertEqual(len(closest_duration_prefix(records, 6.0)), 1)

    def test_epoch_schedule_preserves_baseline_exposure(self):
        schedule = epoch_schedule()
        self.assertEqual(
            {minutes: schedule[f"{minutes}m"]["epochs"] for minutes in TARGET_MINUTES},
            {1: 3000, 5: 600, 15: 200, 30: 100, 60: 50},
        )
        self.assertEqual(schedule["60m"]["diff_epoch"], 10)
        self.assertEqual(schedule["60m"]["joint_epoch"], 30)

    def test_manifest_format(self):
        record = self.utterances[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.txt"
            write_manifest(path, [record], lambda text: f"ipa {text}")
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                f"{record.wav_path}|ipa {record.text}|{record.speaker_id}\n",
            )


if __name__ == "__main__":
    unittest.main()
