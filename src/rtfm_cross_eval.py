"""Cross-domain evaluation of an RTFM checkpoint (the second-backbone arms).

    python src/rtfm_cross_eval.py --target ucf --checkpoint runs/xd_rtfm_E3_s234/RTFM_*.pth
    python src/rtfm_cross_eval.py --target ucf --checkpoint <...> --max-clips 20 --out-dir scratch

Same pooled protocol as ``cross_eval.py`` - target test list in order, batch 1, no
per-video normalisation, scores repeated 16x against the frame GT - but only one
head exists, so the report carries ``AUC1``/``AP1``/``WV_AUC1`` and no ``AUC2``.

This script is what produces the number the paper needs: pooled AUC1 on UCF for a
model trained on XD. Absolute RTFM numbers are expected to be **below** VadCLIP's
(the backbone is weaker); the comparison of record is E3 - E5 within this file's
own backbone, evaluated here for both variants.
"""

import argparse
import json
import os
import platform
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import freq_text_options  # noqa: E402
from cross_eval import gt_path_for, test_list_for  # noqa: E402
from freq_text_dataset import ClipVideoDataset, collate  # noqa: E402
from freq_text_test import GT_FRAME_REPEAT, _f, _i, _within_video_auc  # noqa: E402
from rtfm_eval import frame_scores  # noqa: E402
from rtfm_model import RTFM  # noqa: E402


def run(target, checkpoint, max_clips, out_dir, device, embed_dim, visual_length, tag):
    from torch.utils.data import DataLoader, Subset

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f'checkpoint not found: {checkpoint}')

    gt = np.load(gt_path_for(target)).astype(np.float32)
    dataset = ClipVideoDataset(visual_length, test_list_for(target), test_mode=True,
                               dataset='xd' if target == 'sht' else target)
    if max_clips is not None and max_clips < len(dataset):
        dataset = Subset(dataset, range(max_clips))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)

    model = RTFM(in_dim=embed_dim)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.to(device)

    p1, _names, lens = frame_scores(model, loader, visual_length, device)
    s1 = np.repeat(p1, GT_FRAME_REPEAT)

    expected = len(s1)
    gt_frames_total = _i(len(gt))
    if len(gt) < expected:
        raise ValueError(f'predicted {expected} frames but target GT has {len(gt)}')
    capped = len(gt) != expected
    if capped:
        print(f'  [warn] capped run covered {expected} of {len(gt)} GT frames '
              f'- metrics below are NOT comparable')
        gt = gt[:expected]

    auc1, ap1 = _f(roc_auc_score(gt, s1)), _f(average_precision_score(gt, s1))
    wv1, n_wv = _within_video_auc(p1, lens, gt)

    rep = {
        'backbone': 'RTFM', 'target': target, 'checkpoint': checkpoint,
        'embed_dim': embed_dim, 'visual_length': visual_length,
        'capped': bool(capped), 'n_clips': len(dataset),
        'target_frames': expected, 'gt_frames': gt_frames_total,
        'metrics': {'AUC1': round(auc1, 4), 'AP1': round(ap1, 4),
                    'WV_AUC1': _f(wv1), 'WV_n_videos': n_wv},
        'primary_head': 'AUC1', 'primary_value': round(auc1, 4) if not capped else None,
        'device': device,
        'env': {'python': platform.python_version(), 'torch': torch.__version__},
    }

    out_dir = out_dir or os.path.join(freq_text_options.repo_root(), 'runs')
    os.makedirs(out_dir, exist_ok=True)
    t = tag or 'rtfm'
    suffix = '_smoke' if max_clips is not None else ''
    path = os.path.join(out_dir, f'cross_eval_rtfm_xd2{target}_{t}{suffix}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f'saved -> {path}', flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', required=True, choices=['xd', 'ucf', 'sht'])
    ap.add_argument('--checkpoint', required=True,
                    help='path to an RTFM state dict from xd_train_rtfm.py')
    ap.add_argument('--max-clips', type=int, default=None,
                    help='cap on target videos (smoke only; NOT comparable)')
    ap.add_argument('--out-dir', default=None, help='default runs/; never smoke over runs/')
    ap.add_argument('--tag', default=None, help='result-file tag; defaults to run dir name')
    ap.add_argument('--embed-dim', type=int, default=512)
    ap.add_argument('--visual-length', type=int, default=256)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tag = args.tag or os.path.basename(os.path.dirname(args.checkpoint))
    print(f'[RunCommand] python src/rtfm_cross_eval.py --target {args.target} '
          f'--checkpoint {args.checkpoint} --tag {tag}', flush=True)
    rep = run(args.target, args.checkpoint, args.max_clips, args.out_dir, device,
              args.embed_dim, args.visual_length, tag)
    m = rep['metrics']
    print(f'\nRESULT RTFM xd -> {args.target}')
    print(f'  AUC1 {m["AUC1"]:.4f}  AP1 {m["AP1"]:.4f}  wvAUC1 {m["WV_AUC1"]:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
