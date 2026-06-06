#!/usr/bin/env python3
"""Create reproducible, duration-constrained StyleTTS2 manifests for ESD."""

from __future__ import annotations

import argparse
import json
import random
import wave
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


ENGLISH_SPEAKERS = tuple(f"{speaker:04d}" for speaker in range(11, 21))
EMOTIONS = ("Neutral", "Angry", "Happy", "Sad", "Surprise")
TARGET_MINUTES = (1, 5, 15, 30, 60)
EVAL_SENTENCE_COUNT = 2
BASELINE_MINUTES = 60
BASELINE_EPOCHS = 50


@dataclass(frozen=True)
class Utterance:
    wav_path: str
    source_wav_path: Path
    text: str
    speaker: str
    speaker_id: int
    emotion: str
    base_sentence_id: int
    duration_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create nested 1/5/15/30/60-minute-per-speaker "
            "ESD English manifests."
        )
    )
    parser.add_argument("--esd-root", required=True, help="ESD directory containing wavs.")
    parser.add_argument(
        "--transcript-root",
        required=True,
        help="Original ESD directory containing per-speaker transcript files.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for manifests.")
    parser.add_argument("--path-prefix", default="ESD", help="Manifest wav path prefix.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--language", default="en-us", help="eSpeak language identifier.")
    parser.add_argument(
        "--raw-text",
        action="store_true",
        help="Write source text instead of IPA. Intended only for split inspection.",
    )
    return parser.parse_args()


def base_sentence_id(utterance_id: str) -> int:
    utterance_number = int(utterance_id.rsplit("_", 1)[1])
    return ((utterance_number - 1) % 350) + 1


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def read_utterances(
    esd_root: Path,
    transcript_root: Path,
    path_prefix: str,
    speakers: Sequence[str] = ENGLISH_SPEAKERS,
    emotions: Sequence[str] = EMOTIONS,
) -> list[Utterance]:
    utterances = []
    emotion_set = set(emotions)

    for speaker_id, speaker in enumerate(speakers):
        transcript_path = transcript_root / speaker / f"{speaker}.txt"
        if not transcript_path.is_file():
            raise FileNotFoundError(f"Missing transcript: {transcript_path}")

        with transcript_path.open(encoding="utf-8-sig") as transcript:
            for line_number, line in enumerate(transcript, start=1):
                line = line.rstrip("\r\n")
                if not line:
                    continue
                fields = line.split("\t")
                if len(fields) != 3:
                    raise ValueError(
                        f"Expected 3 tab-separated fields at "
                        f"{transcript_path}:{line_number}"
                    )
                utterance_id, text, emotion = fields
                emotion = emotion.strip()
                if emotion not in emotion_set:
                    continue

                wav_path = esd_root / speaker / emotion / f"{utterance_id}.wav"
                if not wav_path.is_file():
                    raise FileNotFoundError(f"Missing wav: {wav_path}")
                manifest_path = (
                    f"{path_prefix.rstrip('/')}/{speaker}/{emotion}/{utterance_id}.wav"
                )
                utterances.append(
                    Utterance(
                        wav_path=manifest_path,
                        source_wav_path=wav_path,
                        text=text.strip(),
                        speaker=speaker,
                        speaker_id=speaker_id,
                        emotion=emotion,
                        base_sentence_id=base_sentence_id(utterance_id),
                        duration_seconds=wav_duration_seconds(wav_path),
                    )
                )

    expected = len(speakers) * len(emotions) * 350
    if len(utterances) != expected:
        raise ValueError(f"Expected {expected} English ESD utterances, found {len(utterances)}")
    return utterances


def choose_eval_sentence_ids(
    sentence_ids: Iterable[int], seed: int, count: int = EVAL_SENTENCE_COUNT
) -> tuple[list[int], list[int]]:
    shuffled = sorted(set(sentence_ids))
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < count * 2:
        raise ValueError("Not enough sentence IDs for disjoint validation and test sets")
    return sorted(shuffled[:count]), sorted(shuffled[count : count * 2])


def balanced_training_order(
    utterances: Sequence[Utterance],
    excluded_sentence_ids: set[int],
    target_seconds: float,
    seed: int,
) -> list[Utterance]:
    candidates = [
        utterance
        for utterance in utterances
        if utterance.base_sentence_id not in excluded_sentence_ids
    ]
    random.Random(seed).shuffle(candidates)
    queues = defaultdict(deque)
    for utterance in candidates:
        queues[(utterance.speaker, utterance.emotion)].append(utterance)

    speaker_counts: Counter[str] = Counter()
    emotion_counts: Counter[str] = Counter()
    cell_counts: Counter[tuple[str, str]] = Counter()
    sentence_counts: Counter[int] = Counter()
    selected = []
    total_seconds = 0.0
    longest_duration = max(utterance.duration_seconds for utterance in candidates)

    while queues and total_seconds < target_seconds + longest_duration:
        best_cell = min(
            queues,
            key=lambda cell: (
                speaker_counts[cell[0]],
                emotion_counts[cell[1]],
                cell_counts[cell],
                sentence_counts[queues[cell][0].base_sentence_id],
                cell,
            ),
        )
        utterance = queues[best_cell].popleft()
        if not queues[best_cell]:
            del queues[best_cell]
        selected.append(utterance)
        total_seconds += utterance.duration_seconds
        speaker_counts[utterance.speaker] += 1
        emotion_counts[utterance.emotion] += 1
        cell_counts[(utterance.speaker, utterance.emotion)] += 1
        sentence_counts[utterance.base_sentence_id] += 1

    if total_seconds < target_seconds:
        raise ValueError(
            f"Available training audio is only {total_seconds:.2f}s; "
            f"need {target_seconds:.2f}s"
        )
    return selected


def closest_duration_prefix(
    ordered: Sequence[Utterance], target_seconds: float
) -> list[Utterance]:
    total_seconds = 0.0
    for index, utterance in enumerate(ordered):
        previous_seconds = total_seconds
        total_seconds += utterance.duration_seconds
        if total_seconds >= target_seconds:
            if index and abs(previous_seconds - target_seconds) <= abs(
                total_seconds - target_seconds
            ):
                return list(ordered[:index])
            return list(ordered[: index + 1])
    raise ValueError(f"Ordered data does not reach target duration {target_seconds:.2f}s")


def build_splits(
    utterances: Sequence[Utterance], seed: int
) -> tuple[dict[str, list[Utterance]], dict[str, list[int]]]:
    validation_ids, test_ids = choose_eval_sentence_ids(
        (utterance.base_sentence_id for utterance in utterances), seed
    )
    eval_ids = set(validation_ids + test_ids)
    speaker_orders = {
        speaker: balanced_training_order(
            [utterance for utterance in utterances if utterance.speaker == speaker],
            eval_ids,
            max(TARGET_MINUTES) * 60,
            seed,
        )
        for speaker in ENGLISH_SPEAKERS
    }

    splits = {
        f"train_{minutes}m": [
            utterance
            for speaker in ENGLISH_SPEAKERS
            for utterance in closest_duration_prefix(
                speaker_orders[speaker], minutes * 60
            )
        ]
        for minutes in TARGET_MINUTES
    }
    splits["val"] = [
        utterance
        for utterance in utterances
        if utterance.base_sentence_id in validation_ids
    ]
    splits["test"] = [
        utterance
        for utterance in utterances
        if utterance.base_sentence_id in test_ids
    ]
    return splits, {"validation": validation_ids, "test": test_ids}


def epoch_schedule() -> dict[str, dict[str, int]]:
    schedule = {}
    for minutes in TARGET_MINUTES:
        epochs = BASELINE_EPOCHS * BASELINE_MINUTES // minutes
        schedule[f"{minutes}m"] = {
            "epochs": epochs,
            "diff_epoch": max(1, round(epochs * 0.2)),
            "joint_epoch": max(1, round(epochs * 0.6)),
        }
    return schedule


def split_summary(records: Sequence[Utterance]) -> dict[str, object]:
    return {
        "rows": len(records),
        "duration_seconds": round(
            sum(record.duration_seconds for record in records), 6
        ),
        "speakers": dict(sorted(Counter(record.speaker for record in records).items())),
        "speaker_duration_seconds": {
            speaker: round(
                sum(
                    record.duration_seconds
                    for record in records
                    if record.speaker == speaker
                ),
                6,
            )
            for speaker in sorted({record.speaker for record in records})
        },
        "emotions": dict(sorted(Counter(record.emotion for record in records).items())),
        "speaker_emotions": {
            f"{speaker}/{emotion}": count
            for (speaker, emotion), count in sorted(
                Counter((record.speaker, record.emotion) for record in records).items()
            )
        },
        "base_sentence_ids": sorted(
            {record.base_sentence_id for record in records}
        ),
    }


def make_phonemizer(language: str) -> Callable[[str], str]:
    try:
        from phonemizer.backend import EspeakBackend
    except ImportError as error:
        raise RuntimeError(
            "Phonemization requires the `phonemizer` package and eSpeak NG. "
            "Install the repository requirements or pass --raw-text for split inspection."
        ) from error

    backend = EspeakBackend(
        language=language,
        preserve_punctuation=True,
        with_stress=True,
    )
    cache: dict[str, str] = {}

    def phonemize_text(text: str) -> str:
        if text not in cache:
            cache[text] = backend.phonemize([text], strip=True)[0]
        return cache[text]

    return phonemize_text


def write_manifest(
    path: Path,
    records: Sequence[Utterance],
    transform_text: Callable[[str], str],
) -> None:
    with path.open("w", encoding="utf-8") as manifest:
        for record in records:
            text = transform_text(record.text)
            if "|" in text or "\n" in text:
                raise ValueError(f"Manifest text contains a delimiter: {record.text!r}")
            manifest.write(f"{record.wav_path}|{text}|{record.speaker_id}\n")


def validate_splits(splits: dict[str, list[Utterance]]) -> None:
    train_names = [f"train_{minutes}m" for minutes in TARGET_MINUTES]
    previous_paths: set[str] = set()
    for split_name in train_names:
        records = splits[split_name]
        paths = {record.wav_path for record in records}
        if len(paths) != len(records):
            raise ValueError(f"Duplicate wav paths in {split_name}")
        if not previous_paths.issubset(paths):
            raise ValueError(f"{split_name} is not nested with the smaller training split")
        previous_paths = paths

        target_minutes = int(split_name.removeprefix("train_").removesuffix("m"))
        for speaker in ENGLISH_SPEAKERS:
            speaker_records = [
                record for record in records if record.speaker == speaker
            ]
            if not speaker_records:
                raise ValueError(f"{split_name} has no data for speaker {speaker}")
            emotion_counts = Counter(
                record.emotion for record in speaker_records
            )
            if set(emotion_counts) != set(EMOTIONS):
                raise ValueError(
                    f"{split_name} does not cover all emotions for speaker {speaker}"
                )
            if max(emotion_counts.values()) - min(emotion_counts.values()) > 1:
                raise ValueError(
                    f"{split_name} is emotion-imbalanced for speaker {speaker}"
                )

            duration = sum(
                record.duration_seconds for record in speaker_records
            )
            tolerance = max(
                record.duration_seconds for record in speaker_records
            ) / 2
            if abs(duration - target_minutes * 60) > tolerance:
                raise ValueError(
                    f"{split_name} misses the duration target for speaker {speaker}"
                )

    train_paths = {record.wav_path for record in splits[train_names[-1]]}
    val_paths = {record.wav_path for record in splits["val"]}
    test_paths = {record.wav_path for record in splits["test"]}
    if train_paths & val_paths or train_paths & test_paths or val_paths & test_paths:
        raise ValueError("Train, validation, and test wav paths must be disjoint")

    train_sentence_ids = {
        record.base_sentence_id for record in splits[train_names[-1]]
    }
    val_sentence_ids = {record.base_sentence_id for record in splits["val"]}
    test_sentence_ids = {record.base_sentence_id for record in splits["test"]}
    if (
        train_sentence_ids & val_sentence_ids
        or train_sentence_ids & test_sentence_ids
        or val_sentence_ids & test_sentence_ids
    ):
        raise ValueError("Train, validation, and test transcript IDs must be disjoint")

    expected_eval_rows = (
        EVAL_SENTENCE_COUNT * len(ENGLISH_SPEAKERS) * len(EMOTIONS)
    )
    for split_name in ("val", "test"):
        if len(splits[split_name]) != expected_eval_rows:
            raise ValueError(
                f"{split_name} must contain {expected_eval_rows} rows, "
                f"found {len(splits[split_name])}"
            )
        cell_counts = Counter(
            (record.speaker, record.emotion) for record in splits[split_name]
        )
        if set(cell_counts.values()) != {EVAL_SENTENCE_COUNT}:
            raise ValueError(f"{split_name} is not balanced by speaker and emotion")


def main() -> None:
    args = parse_args()
    utterances = read_utterances(
        Path(args.esd_root), Path(args.transcript_root), args.path_prefix
    )
    splits, eval_sentence_ids = build_splits(utterances, args.seed)
    validate_splits(splits)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transform_text = (lambda text: text) if args.raw_text else make_phonemizer(args.language)
    filenames = {
        **{
            f"train_{minutes}m": f"train_list_esd_{minutes}m.txt"
            for minutes in TARGET_MINUTES
        },
        "val": "val_list_esd.txt",
        "test": "test_list_esd.txt",
    }
    for split_name, filename in filenames.items():
        write_manifest(out_dir / filename, splits[split_name], transform_text)

    metadata = {
        "seed": args.seed,
        "language": args.language,
        "phonemized": not args.raw_text,
        "path_prefix": args.path_prefix,
        "eval_sentence_ids": eval_sentence_ids,
        "epoch_schedule": epoch_schedule(),
        "splits": {
            split_name: split_summary(records)
            for split_name, records in splits.items()
        },
    }
    with (out_dir / "split_metadata.json").open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2, sort_keys=True)
        output.write("\n")

    for split_name, records in splits.items():
        duration = sum(record.duration_seconds for record in records)
        print(f"{split_name}: {len(records)} rows, {duration / 60:.3f} minutes")


if __name__ == "__main__":
    main()
