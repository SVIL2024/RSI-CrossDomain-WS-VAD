# Structure, Not Semantics — code for source-only cross-domain WS-VAD

Code for the paper **"Structure, Not Semantics: What Transfers in Source-Only Cross-Domain
Weakly-Supervised Video Anomaly Detection"**.

## Overview

This repository contains the code for rank-1 spectral injection (RSI), a lightweight
source-only augmentation for cross-domain weakly-supervised video anomaly detection (WS-VAD)
on frozen CLIP features. During MIL training the method forms a rank-1 update of the
lowest-scoring frames of source-normal features in the frequency domain and injects it
through a normalised, RMS-matched spectral field.

The paper's contribution is an attribution study rather than a new state of the art. It
isolates what actually transfers: the *structure* of the field — a rank-1, RMS-matched
perturbation aimed at the hardest source frames — and not the semantic content of the
direction or the shape of the gate. "Structure" here means rank-1 sharing, not smoothness by
itself: an equally smooth field of full rank returns the baseline. The same field is actively
harmful on a second backbone (RTFM), reported as a negative result rather than hidden.

## Repository structure

```
├── src/
│   ├── model.py                  # CLIPVAD dual-head model (VadCLIP architecture)
│   ├── freq_text_aug.py          # RSI augmenter (shift/band/timedomain/noise modes)
│   ├── freq_text_options.py      # Variant table + CLI arguments + data-root helpers
│   ├── freq_text_prompts.py      # Class definitions + prompt templates (XD/UCF/union)
│   ├── freq_text_dataset.py      # Dataset loader with 16-frame stride convention
│   ├── freq_text_trainer.py      # Training loop (CLAS2 + CLASM + dispersion + aug)
│   ├── freq_text_test.py         # Evaluation (pooled AUC/AP + within-video AUC)
│   ├── freq_text_text.py         # CLIP text direction bank builder
│   ├── freq_text_events.py       # Event-prompt dictionary + per-frame response matrix
│   ├── probe_events.py           # Event-prompt / direction probe
│   ├── xd_option.py              # XD-Violence options
│   ├── xd_train.py               # XD-Violence training entrypoint
│   ├── xd_test.py                # XD-Violence in-domain evaluation entrypoint
│   ├── ucf_option.py             # UCF-Crime options
│   ├── ucf_train.py              # UCF-Crime training entrypoint (UCF->XD row)
│   ├── ucf_test.py               # UCF-Crime in-domain evaluation entrypoint
│   ├── union_train.py            # Multi-source (XD+UCF) training entrypoint
│   ├── build_union_list.py       # Merged XD+UCF training list builder
│   ├── cross_eval.py             # Cross-domain evaluation (source ckpt -> target test)
│   ├── rtfm_model.py             # RTFM network (second backbone, 17.9M params)
│   ├── rtfm_eval.py              # RTFM frame scoring + pooled/within-video metrics
│   ├── xd_train_rtfm.py          # RTFM training entrypoint (same recipe as xd_train.py)
│   ├── rtfm_cross_eval.py        # RTFM cross-domain evaluation
│   ├── seed_stats.py             # Aggregates runs/cross_eval_*.json into per-arm statistics
│   ├── six_seed_stats.py        # Reproduces every number quoted in the paper's tables
│   ├── precondition_test.py      # Precondition-probe diagnostic
│   ├── analysis_crossdomain_transfer.py  # Direction transferability diagnostic
│   ├── analysis_spectral_bands.py         # Temporal band diagnostic (H2)
│   ├── analysis_zsad_floor.py             # Training-free ZSAD floor measurement
│   ├── analysis_official_metric.py        # Published-protocol trivial baselines
│   ├── analysis_gain_vs_difficulty.py     # Gain vs. difficulty stratification
│   ├── analysis_magnitude_orthogonal.py   # Magnitude-vs-direction orthogonality
│   ├── analysis_residual_direction.py     # Residual-direction attribution
│   ├── analysis_selective_routing.py     # Selective-routing diagnostic
│   ├── analysis_selective_trust.py        # Selective-trust diagnostic
│   ├── analysis_spread_confound.py        # Score-spread confound check
│   ├── stage0_extract_smoke.py   # CLIP feature extraction smoke test
│   ├── sht_extract.py            # ShanghaiTech CLIP feature extraction
│   ├── sht_build_gt.py           # ShanghaiTech GT construction
│   ├── clip/                     # Vendored CLIP (from VadCLIP), incl. the BPE vocab
│   └── utils/                    # Vendored utilities (from VadCLIP)
├── list/                         # Training/test lists + ground truth
│   ├── xd_CLIP_rgb.csv           # XD-Violence training list
│   ├── xd_CLIP_rgbtest.csv       # XD-Violence test list
│   ├── ucf_CLIP_rgb.csv          # UCF-Crime training list
│   ├── ucf_CLIP_rgbtest.csv      # UCF-Crime test list
│   ├── union_CLIP_rgb.csv        # XD+UCF merged training list (multi-source)
│   ├── gt.npy / gt_ucf.npy       # Frame-level ground truth
│   ├── gt_segment.npy / ...      # Segment-level GT
│   └── repair_list_paths.py      # Re-root the absolute feature paths stored in the CSVs
├── tests/
│   ├── test_freq_text_aug.py     # Unit tests for the augmenter
│   ├── test_training_protocol.py # Training-protocol regression tests
│   └── test_public_paths.py      # Entrypoint imports + path-helper API guard
├── run_ablation.py               # Crash-safe ablation driver (the R1--R4 sweep)
├── LICENSE
├── requirements.txt
└── README.md
```

## Requirements

Python 3.9+; `pip install -r requirements.txt` (PyTorch 1.13+ with CUDA recommended, numpy, scikit-learn, scipy, pandas, ftfy, regex, Pillow, opencv-python, tqdm). OpenAI CLIP is vendored in `src/clip/`.

## Data

Use the released VadCLIP CLIP ViT-B/16 frame features (stride 16, 512-d fp16, one feature per 16 frames): <https://github.com/nwpu-zxr/VadCLIP>. Lay them out as

```
<data root>/
├── XDTrainClipFeatures/
├── XDTestClipFeatures/
├── UCFTrainClipFeatures/
├── UCFTestClipFeatures/
└── SHTClipFeatures/          # only needed for the ShanghaiTech target
```

All path resolution goes through `freq_text_options`: `data_root()`, `sht_root()`, `sht_feat_root()` and `official_checkpoints(source)`. Each honours the same precedence:

1. the `RSI_*` variable (current name)
2. the `FGDA_*` variable (pre-rename name, still honoured for existing setups)
3. a built-in default

| Variable | Controls | Default |
|---|---|---|
| `RSI_DATA_ROOT` | directory **containing** the feature banks above | `E:/dataset` |
| `RSI_SHT_ROOT` | raw ShanghaiTech frames | `E:\dataset\shanghaitech` |
| `RSI_SHT_FEAT_ROOT` | extracted ShanghaiTech features | `<data root>/SHTClipFeatures` |
| `RSI_VADCLIP_XD` / `RSI_VADCLIP_UCF` | released VadCLIP checkpoints | `<repo>/checkpoints/model_{xd,ucf}.pth`, then two development-machine paths |

An empty-string variable counts as unset, so you can blank a default in CI without deleting the export. `tests/test_public_paths.py` pins all of this.

The list CSVs were produced on the machine that built the feature banks and store absolute paths (`E:/dataset/...`). **Do not edit 55k rows by hand** — set `RSI_DATA_ROOT` to the directory above (the one containing `XDTrainClipFeatures`) and the stored paths are re-rooted automatically at load time. If your directory names differ from the originals, `python list/repair_list_paths.py --roots <dir>` rewrites each list by locating the `.npy` basename (a `.bak` is kept); `--check-only` reports without writing.

Official VadCLIP checkpoints are needed for `--checkpoint official`, which is also how the released UCF->XD row is produced. Download `model_xd.pth` / `model_ucf.pth` from the VadCLIP repository, then either set `RSI_VADCLIP_XD` / `RSI_VADCLIP_UCF`, or drop them at `checkpoints/model_xd.pth` / `checkpoints/model_ucf.pth` inside the clone.

ShanghaiTech (target only): extract features with `python src/sht_extract.py --split both`, then build the test list and frame GT with `python src/sht_build_gt.py`.

## Usage

### Train on XD-Violence

```bash
# E5 baseline (no augmentation)
python src/xd_train.py --variant E5 --seed 234

# E3: RSI with a fixed random rank-1 direction (the method)
python src/xd_train.py --variant E3 --seed 234 --save-threshold 0

# E1: RSI with the CLIP text class direction instead (ablated)
python src/xd_train.py --variant E1 --seed 234 --save-threshold 0

# E3a / E3b: the gate ablation of the paper's Table 5
python src/xd_train.py --variant E3a --seed 234 --save-threshold 0
python src/xd_train.py --variant E3b --seed 234 --save-threshold 0

# N1: matched-magnitude noise control
python src/xd_train.py --variant N1 --seed 234 --save-threshold 0
```

Artefacts of a variant land in `runs/<dataset>_<variant>/`. The best model is written both under a self-describing filename (`XD_auc..._ap..._e{N}_...pth`) and as the stable alias `runs/<dataset>_<variant>/model_best.pth`; `latest_best.txt` also records the self-describing name.

### Train on UCF-Crime (source side of the UCF->XD row)

```bash
python src/ucf_train.py --variant E5 --seed 234
python src/ucf_train.py --variant E3 --seed 234 --save-threshold 0
```

Note the metric asymmetry that the paper's protocol section states explicitly: on XD the
headline metric is AP, on UCF it is AUC. `--save-threshold` and the checkpoint naming follow
that choice, so a UCF checkpoint is written as `UCF_auc..._ap..._e{N}_...pth`.

### In-domain evaluation

```bash
python src/xd_test.py --variant E3      # loads runs/xd_E3/model_best.pth
python src/ucf_test.py --variant E3      # loads runs/ucf_E3/model_best.pth
```

Both wrappers resolve the checkpoint from `runs/<dataset>_<variant>/model_best.pth`; pass the
same `--variant` you trained with.

### Cross-domain evaluation

```bash
# Evaluate an XD-trained checkpoint on UCF-Crime
python src/cross_eval.py --source xd --target ucf --checkpoint runs/xd_E3/model_best.pth

# Evaluate the official VadCLIP checkpoint on UCF-Crime
python src/cross_eval.py --source xd --target ucf --checkpoint official

# UCF-to-XD row (official UCF checkpoint)
python src/cross_eval.py --source ucf --target xd --checkpoint official
```

### Second backbone: RTFM (paper Section 4.5)

`rtfm_model.py` is a re-implementation of the RTFM network that runs on the same feature banks
and the same training recipe as VadCLIP. The original RTFM code hard-codes a
`batch x 10 crops x T x 512` input layout and a fixed 32-segment grid; this port keeps the
architecture (17.9M parameters) but feeds it the same variable-length batches as `xd_train.py`,
so the only difference between the two blocks of the paper's second-backbone table is the
backbone itself. RTFM has a single head and selects its top-k abnormal segments by magnitude,
so the same `--variant` flags (E5, E3, N1, ...) apply unchanged.

```bash
# Baseline and RSI field on RTFM; 30 epochs, milestones at 10/20/28
python src/xd_train_rtfm.py --variant E5 --seed 234 --max-epoch 30 --scheduler-milestones 10 20 28
python src/xd_train_rtfm.py --variant E3 --seed 234 --max-epoch 30 --scheduler-milestones 10 20 28

# Cross-domain evaluation of an RTFM checkpoint
python src/rtfm_cross_eval.py --target ucf --checkpoint runs/xd_rtfm30_E3_s234/<ckpt>.pth
```

The 30-epoch budget is not optional. With the shared VadCLIP schedule (`[3, 6, 10]`) the
learning rate is already down to 1e-6 by epoch 3, which leaves RTFM undertrained and makes the
comparison meaningless; all results in the paper use 30 epochs for both arms. RTFM's own
baseline also varies far more across seeds than VadCLIP's (8.2 pooled AUC1 points over three
seeds), which is why the paper reports the negative result as a sign rather than a magnitude.

### Seed statistics

`six_seed_stats.py` reproduces every mean, standard deviation, paired difference, exact
one-sided sign test and percentile bootstrap interval quoted in the paper. Its input is the
per-seed pooled AUC1 values printed by `cross_eval.py`, hard-coded so the file doubles as a
record of what was run:

```bash
python src/six_seed_stats.py
```

`seed_stats.py` does the same aggregation directly from `runs/cross_eval_*.json`, which is what
you want after a fresh campaign:

```bash
python src/seed_stats.py --seeds 234,555,789,222,999
python src/seed_stats.py --drop-seeds 111 --out runs/stats.md
```

### Multi-source (union) training

```bash
# Build the merged XD+UCF training list
python src/build_union_list.py

# Train on the union list
python src/union_train.py --variant E3 --seed 234

# Evaluate union-trained model on held-out SHT
python src/cross_eval.py --source union --target sht --checkpoint runs/union_E3/model_best.pth
```

### Diagnostics

```bash
# Direction transferability probe
python src/analysis_crossdomain_transfer.py

# Spectral band diagnostic
python src/analysis_spectral_bands.py

# Training-free ZSAD floor
python src/analysis_zsad_floor.py

# Published-protocol trivial baselines
python src/analysis_official_metric.py

# Attribution diagnostics backing the paper's analysis section
python src/analysis_gain_vs_difficulty.py     # gain vs. difficulty stratification
python src/analysis_magnitude_orthogonal.py   # magnitude vs. direction orthogonality
python src/analysis_residual_direction.py     # residual-direction attribution
python src/analysis_selective_routing.py     # selective-routing diagnostic
python src/analysis_selective_trust.py        # selective-trust diagnostic
python src/analysis_spread_confound.py        # score-spread confound check
python src/precondition_test.py               # precondition-probe diagnostic
```

### Tests

```bash
python -m pytest tests/ -q      # 96 tests: augmenter units, training protocol, path API
```

## Variants

| ID | Description |
|---|---|
| E5 | Baseline (no augmentation, VadCLIP as published) |
| E3 | RSI with a fixed random rank-1 direction -- **the method** |
| E1 | RSI with the CLIP text class direction (ablated) |
| E3a | E3 with the designed gate frozen at initialisation |
| E3b | E3 with a frozen random spectrum (destroys the designed band shape) |
| N1 | Plain i.i.d. noise control (RMS-matched, no structure) |
| E3c | E3 with the basis block drawn uniformly at random instead of by score (`--aug-frame-select random`) |
| E3w | E3 with the basis block as one contiguous run centred on the hardest frame (`--aug-frame-select contiguous`) |
| N2 | Temporally smooth **full-rank** noise: each feature dimension gets its own band-limited field (same envelope as E3, rank-1 sharing destroyed) |
| B1 | Multiplicative band gain (non-degenerate spectral form) |
| E4 | Gate frozen flat (degenerate) |
| E2 | Time-domain fallback (degenerate) |

## Refinement sweep (paper Table 4): reproducing R1--R4

R1--R4 are the E3 method with one knob changed. There are no separate variant IDs; pass the override on top of `--variant E3` (CLI flags take precedence over the variant table):

| Paper row | Command |
|---|---|
| E3 default | `--variant E3` (rho=0.2, K=51, alpha in [0.1,0.5]) |
| R4 band mode | `--variant B1` (multiplicative band gain, identity-init gate) |
| R3 sparse high-freq | `--variant E3 --aug-ratio 0.09 --aug-block-len 25` |
| R1 coverage+ | `--variant E3 --aug-ratio 0.35 --aug-block-len 90` |
| R2 severity+ | `--variant E3 --aug-alpha-lo 0.3 --aug-alpha-hi 0.7` |

`python run_ablation.py` drives the whole sweep crash-safely (it restarts only the runs that did not finish, so it is safe to re-invoke after an interruption).

## Key findings

1. Over six seeds the field moves pooled AUC on XD->UCF from 0.7766 +- 0.0165 to
   0.7919 +- 0.0179, a paired gain of **+1.52 +- 0.91** (6/6 seeds, exact one-sided sign test
   p = 0.016, bootstrap 95% CI [+0.85, +2.17]); a video-level bootstrap over the 290 test
   videos gives [+0.46, +2.66], and the interval for any single run is two to three points
   wide, i.e. wider than the effect itself. The sixth seed (777) was fixed before it was run and reported whatever it showed.
   The gain comes from the **injection scaffold** -- an RMS-matched spectral field aimed at
   the hardest source frames -- and not from the semantic content of the direction.
2. CLIP text class directions are close to useless as perturbation directions: +0.69 +- 1.24
   over E5 (CI [-0.03, +1.72]), and the field built from a random rank-1 direction beats the
   text-direction field by +0.83 +- 0.99 (5/6). The advantage is a matter of degree, not of
   kind: both variants gain, and what separates them is that E1 gains mostly on weak baselines
   (correlation between baseline AUC and gain -0.77) while E3's gain is independent of baseline
   strength (-0.12). The assumption that text directions carry transferable anomaly semantics
   does not survive the attribution.
3. What the controls establish is narrower than what the mechanism was named for. Destroying
   the lowest-score placement rule leaves +0.79 +- 1.12 of the gain (+0.73 +- 0.95 as a paired
   difference against the full method, 4/6, p = 0.34); replacing the shared rank-1 direction
   with per-dimension noise of the same envelope and energy (N2) is not separable
   (+0.82 +- 2.35 against it, 4/6, p = 0.34) but is far less stable (spread 2.35 against 0.91)
   and no better than the baseline (+0.71 +- 2.75); a contiguous window instead of the rank
   gathering is also not separable (+0.35 +- 0.86). So no single component is proved to carry
   the effect. What the data do support is reliability: E3 is the only variant whose gain
   survives every seed and does not come from rescuing weak baselines.
4. The designed gate is **not** the active ingredient: with three seeds, freezing the gate
   leaves +1.12 +- 0.57 and a frozen random spectrum leaves +1.36 +- 0.76, both inside the
   spread of the three-run estimate of the full method, so the sweep bounds sensitivity rather
   than locating an optimum.
5. The transfer is not XD/UCF-specific: training on UCF and evaluating on XD moves AP2 by
   +1.66 over the released UCF baseline (66.14 vs 64.48); on the class-agnostic head the same
   direction loses -3.24 (57.83 -> 54.59), so the two heads disagree in sign and no gain is
   claimed there.
6. The effect does **not** generalise to a second backbone: on RTFM (30 epochs, three seeds)
   the same field costs -7.9 pooled AUC1 (range -3.8 to -11.8), because RTFM's single head
   selects its top-k segments by magnitude and the energy-matched field corrupts exactly that
   statistic, while VadCLIP's second, text-anchored head does not.
7. The hardest target (ShanghaiTech) exhibits a between-video style shift that dominates the
   pooled metric: a trivial magnitude cue reaches 0.86 within-video AUC while pooling below
   chance, and every arm sits below chance in both protocols.

## Citation

If you use this code, please cite the paper.

## License

MIT (see `LICENSE`).
