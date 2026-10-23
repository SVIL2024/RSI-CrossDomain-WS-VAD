"""Seed-level statistics for the XD -> UCF arms.

Reads every ``runs/cross_eval_xd2ucf_*.json`` written by ``cross_eval.py`` /
``rtfm_cross_eval.py``, recovers ``(variant, seed)`` from the **checkpoint file
name** (never from the json tag, which is free-form), and prints

* the per-seed pooled AUC1 matrix,
* per-arm mean / sample sd,
* the paired difference against E5 on the seeds the two arms share,
* an exact sign test and a bootstrap CI for the mean paired difference.

Why the checkpoint name: a run directory may be recycled (``runs/xd_E5`` holds
seed 234) and a json tag is arbitrary (``gate_E5_s234``, ``xd_E5_s234``,
``noise_n1`` all exist for the same arm). The checkpoint name carries
``variant=..._seed=...`` verbatim from the training command line and cannot lie.

Arms with a different backbone (``cross_eval_rtfm_*``) are reported separately:
an E3-E5 difference is only interpretable within one backbone.

    python src/seed_stats.py                 # markdown to stdout
    python src/seed_stats.py --out stats.md
"""

import argparse
import glob
import json
import os
import re
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', 'runs')

# Tag prefixes that hold the arms of record. Everything else in runs/ is a
# probe, a refinement sweep or a smoke test and must not enter the statistics.
KEEP = ('gate_', 'noise_', 'n2_', 'e3c_', 'e3w_', 'ext_', 'xd_')
PAT = re.compile(r'variant=([A-Za-z0-9]+)_seed=(\d+)')
ORDER = ['E5', 'E1', 'E3', 'E3a', 'E3b', 'E3c', 'E3w', 'N1', 'N2', 'B1']


def collect(pattern, root=ROOT):
    """Return ``{(variant, seed): (auc1, ap1, filename)}`` for one backbone."""
    recs = {}
    for path in sorted(glob.glob(os.path.join(root, pattern))):
        base = os.path.basename(path)
        tag = base.split('cross_eval_', 1)[-1]
        tag = tag.split('xd2ucf_', 1)[-1] if 'xd2ucf_' in tag else tag
        tag = tag.split('xd2sht_', 1)[-1] if 'xd2sht_' in tag else tag
        if not tag.startswith(KEEP) or 'smoke' in base:
            continue
        try:
            d = json.load(open(path, encoding='utf-8'))
        except json.JSONDecodeError:
            continue
        if d.get('target') != 'ucf':
            continue
        m = PAT.search(os.path.basename(str(d.get('checkpoint', ''))))
        if not m:
            continue
        key = (m.group(1), int(m.group(2)))
        # xd_* is the campaign run; gate_* re-decoded the same checkpoints. Keep
        # one record per (variant, seed), preferring the campaign file.
        prio = 0 if tag.startswith('xd_') else 1
        if key not in recs or prio < recs[key][0]:
            met = d.get('metrics', {})
            recs[key] = (prio, met.get('AUC1'), met.get('AP1'), base)
    return {k: v[1:] for k, v in recs.items()}


def sign_test(diffs):
    """One-sided exact binomial p for "the difference is positive"."""
    n = len(diffs)
    k = sum(1 for d in diffs if d > 0)
    if n == 0:
        return float('nan'), 0, 0
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n, k, n


def bootstrap_ci(diffs, reps=20000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    means = rng.choice(d, size=(reps, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def table(recs, title):
    arms = [a for a in ORDER if any(k[0] == a for k in recs)]
    if not arms:
        return f'## {title}\n\n(no results yet)\n'
    seeds = sorted({k[1] for k in recs})
    base = 'E5'
    base_map = {s: recs[(base, s)][0] for s in seeds if (base, s) in recs}

    out = [f'## {title}', '',
           '| arm | ' + ' | '.join(f's{s}' for s in seeds) + ' | mean | sd | d(E5) | +/n | sign p | 95% CI |',
           '|' + '---|' * (len(seeds) + 7)]
    for a in arms:
        cells, vals, diffs = [], [], []
        for s in seeds:
            if (a, s) in recs:
                v = recs[(a, s)][0]
                cells.append(f'{v:.4f}'.lstrip('0'))
                vals.append(v)
                if s in base_map:
                    diffs.append((v - base_map[s]) * 100)
            else:
                cells.append('-')
        mean = float(np.mean(vals)) if vals else float('nan')
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float('nan')
        if a == base or not diffs:
            dstr = 'base' if a == base else '-'
            pstr, ci, pos = '-', '-', '-'
        else:
            p, k, n = sign_test(diffs)
            lo, hi = bootstrap_ci(diffs)
            dstr = f'{np.mean(diffs):+.2f}±{np.std(diffs, ddof=1):.2f}'
            pstr = f'{p:.3f}'
            ci = f'[{lo:+.2f}, {hi:+.2f}]'
            pos = f'{k}/{n}'
        out.append(f'| {a} | ' + ' | '.join(cells) +
                   f' | {mean:.4f} | {sd:.4f} | {dstr} | {pos} | {pstr} | {ci} |')
    out.append('')
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=None, help='write markdown here instead of stdout')
    ap.add_argument('--seeds', default=None,
                    help='comma-separated seed whitelist, e.g. 234,555,789')
    ap.add_argument('--drop-seeds', default=None,
                    help='comma-separated seeds to leave out, e.g. 111')
    args = ap.parse_args()

    keep = {int(s) for s in args.seeds.split(',')} if args.seeds else None
    drop = {int(s) for s in args.drop_seeds.split(',')} if args.drop_seeds else set()

    def filt(recs):
        return {k: v for k, v in recs.items()
                if k[1] not in drop and (keep is None or k[1] in keep)}

    v = filt(collect('cross_eval_xd2ucf_*.json'))
    r = filt(collect('cross_eval_rtfm_xd2ucf_*.json'))
    md = ('# XD -> UCF, pooled AUC1\n\n'
          'Differences are in AUC points, paired on shared seeds. `sign p` is the '
          'one-sided exact binomial p for "E3-style arm above E5"; `95% CI` is a '
          'percentile bootstrap over the paired differences (20k resamples, fixed '
          'seed), which with n<=5 is indicative only.\n\n')
    md += table(v, 'VadCLIP backbone')
    md += '\n' + table(r, 'RTFM backbone')
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f'wrote {args.out}')
    else:
        print(md)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
