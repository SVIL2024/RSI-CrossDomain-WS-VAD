"""Pooled-protocol evaluation for the RTFM backbone.

The paper's numbers are **pooled** AUC/AP over every frame of every video in test
list order, scores repeated 16x to match the frame-level GT, no per-video
normalisation (``freq_text_test`` calls this the "published" protocol; the
``WV_*`` numbers are the within-video variant and must never be mixed in).

Two things differ from ``freq_text_test`` and both are deliberate:

* **No second head.** RTFM has no text branch, so there is no ``logits2`` and no
  ``det_mAP`` (that metric needs per-class probabilities). Only head1 exists:
  ``AUC1``/``AP1``/``WV_AUC1``.
* **Explicit windowing.** ``_frame_probs`` relies on ``CLIPVAD`` reshaping a
  ``(L, C)`` tensor into ``maxlen``-sized windows internally with a padding mask.
  RTFM has no such reshape, so the windowing is done here: a video longer than
  ``maxlen`` is cut into consecutive non-overlapping windows, each scored at full
  length, and the scores are concatenated in order. For the videos in XD/UCF that
  fit in one 256-frame window - all of them at ``visual_length=256`` - this is the
  same pass as ``_frame_probs`` minus the padding, which cannot change a
  comparison between two RTFM runs.
"""

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from freq_text_test import GT_FRAME_REPEAT, _f, _i, _within_video_auc

__all__ = ['frame_scores', 'make_rtfm_test_fn']


def frame_scores(model, loader, maxlen, device):
    """Run the split-window pass; return ``(probs (N,), names, lens)``.

    ``probs`` is one sigmoid anomaly score per frame, concatenated across videos
    in loader order - the same ordering ``_frame_probs`` produces, so the
    downstream ``np.repeat(probs, 16)`` lines up with the GT array.
    """
    model.to(device)
    model.eval()
    out, names, lens = [], [], []

    with torch.no_grad():
        for idx, item in enumerate(loader):
            # collate emits (1, T, C) on some paths and (1, 1, T, C) on others
            # (VadCLIP's own test loop squeezes exactly one dim and then
            # conditionally adds one back). Flattening every leading singleton
            # dimension is shape-proof for both, and (T, C) is all this pass needs.
            raw = item[0]
            feat = raw.reshape(-1, raw.shape[-1])
            length = min(_i(item[2]), int(feat.shape[0]))

            chunks = []
            for start in range(0, max(length, 1), maxlen):
                chunk = feat[start:start + maxlen]
                if chunk.shape[0] == 0:
                    continue
                logits, _ = model(chunk.unsqueeze(0).to(device))
                chunks.append(torch.sigmoid(logits.reshape(-1).float()).cpu())
            probs = torch.cat(chunks)[:length] if chunks else torch.zeros(0)

            out.append(probs)
            names.append(str(item[1][0]))
            lens.append(length)
            if idx % 25 == 0 or length > maxlen:
                print(f'    [eval] {idx:4d} {str(item[1][0])[:40]:40s} raw={tuple(raw.shape)} '
                      f'len={length} windows={len(chunks)}', flush=True)

    if not out:
        raise ValueError('test loader produced no videos - nothing to score')
    return torch.cat(out).numpy(), names, lens


def make_rtfm_test_fn(dataset):
    """Build ``test(model, loader, maxlen, _prompt, gt, gtseg, gtlabel, device)``.

    Signature-compatible with ``freq_text_test.make_test_fn`` so the training loop
    can call either one; ``_prompt``/``gtseg``/``gtlabel`` are accepted and ignored
    (no text head, no detection mAP).

    Returns ``(primary_metric, extra)``: ``primary`` is AP for XD (that is what
    VadCLIP selects checkpoints on for XD) and pooled AUC1 for UCF (the paper's
    cross-domain headline). ``extra`` always carries ``AUC1``/``AP1``/``WV_AUC1``.
    """

    def test(model, loader, maxlen, _prompt, gt, _gtsegments, _gtlabels, device):
        p1, _names, lens = frame_scores(model, loader, maxlen, device)
        s1 = np.repeat(p1, GT_FRAME_REPEAT)

        gt = np.asarray(gt)
        expected = len(s1)
        if len(gt) < expected:
            raise ValueError(f'predicted {expected} frames but GT only has {len(gt)}')
        capped = len(gt) != expected
        if capped:
            print(f'    [warn] capped run covered {expected} of {len(gt)} GT frames '
                  f'- metrics below are NOT comparable')
            gt = gt[:expected]

        auc1 = roc_auc_score(gt, s1)
        ap1 = average_precision_score(gt, s1)
        wv1, n_wv = _within_video_auc(p1, lens, gt)

        extra = {'AUC1': _f(auc1), 'AP1': _f(ap1),
                 'WV_AUC1': wv1, 'WV_n_videos': n_wv, 'capped': bool(capped)}
        primary = ap1 if dataset == 'xd' else auc1
        print(f'    AUC1 {auc1:.4f} AP1 {ap1:.4f} | wvAUC1 {wv1:.4f} (n={n_wv})')
        return _f(primary), extra

    return test
