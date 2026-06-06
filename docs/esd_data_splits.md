# ESD minute-budget splits

`scripts/create_esd_splits.py` creates controlled English ESD fine-tuning
manifests for 1, 5, 15, 30, and 60 minutes of speech per speaker. With all 10
English speakers, the combined multispeaker manifests contain approximately
10, 50, 150, 300, and 600 minutes.

## Split policy

ESD has 350 parallel base sentences for each of 10 English speakers and five
emotions. The official corpus protocol commonly reserves complete base
sentence IDs, which prevents the same transcript appearing in training and
evaluation.

The generator reserves two seeded base sentence IDs for validation and two
different IDs for test. Each evaluation split therefore has 100 utterances:
two for every speaker-emotion combination. This is the smallest practical
fixed evaluation set with repeated coverage of all 50 combinations.

Training candidates exclude every evaluation transcript ID. For each speaker,
a seeded greedy ordering favors the currently least represented emotion and
base sentence. Each speaker's duration budgets are prefixes of one ordering,
and the combined multispeaker manifests are therefore nested. The prefix
immediately above or below each per-speaker target is selected according to
which is closer in duration.

The referenced `StyleTTS2FineTune` phonemizer settings are retained:
eSpeak `en-us`, punctuation preservation, and lexical stress. Its sequential
90/10 split is not used because ESD utterance IDs are grouped by emotion and
that approach is neither balanced nor duration-controlled.

## Generate manifests

The validated seed-1234 manifests and their metadata are committed in
`Data/`. Regenerate them with the command below when the source corpus or
split policy changes.

Run this where both the processed wav tree and original transcript tree are
available:

```bash
python3 scripts/create_esd_splits.py \
  --esd-root /store/store2/data/ESD \
  --transcript-root /store/store2/data/ESD_og \
  --out-dir Data \
  --path-prefix ESD \
  --seed 1234
```

The command writes `train_list_esd_{1,5,15,30,60}m.txt`, fixed
`val_list_esd.txt` and `test_list_esd.txt` files, and
`split_metadata.json`. Metadata records total and per-speaker durations,
balance counts, base sentence IDs, and training schedules. Use `--raw-text`
only to inspect split selection without a phonemizer/eSpeak installation.

## Training schedule

The 60-minute-per-speaker multispeaker baseline uses 50 epochs, with diffusion
starting at epoch 10 and joint training at epoch 30. To keep the approximate
number of optimizer updates comparable across data budgets, scale epochs
inversely with per-speaker minutes:

| Data per speaker | Combined data | Epochs | Diffusion epoch | Joint epoch |
| ---: | ---: | ---: | ---: | ---: |
| 1 minute | 10 minutes | 3000 | 600 | 1800 |
| 5 minutes | 50 minutes | 600 | 120 | 360 |
| 15 minutes | 150 minutes | 200 | 40 | 120 |
| 30 minutes | 300 minutes | 100 | 20 | 60 |
| 60 minutes | 600 minutes | 50 | 10 | 30 |

Actual updates can differ slightly because each duration target ends at a
whole utterance and the final batch may be partial. Compare runs by optimizer
step as well as epoch, and select checkpoints using the same fixed validation
manifest.

## References

- [Emotional Voice Conversion: Theory, Databases and ESD](https://arxiv.org/abs/2105.14762)
- [ESD dataset repository](https://github.com/HLTSingapore/Emotional-Speech-Data)
- [StyleTTS2FineTune phonemization guide](https://github.com/IIEleven11/StyleTTS2FineTune)
