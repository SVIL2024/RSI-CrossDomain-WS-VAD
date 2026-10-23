"""XD training with the RTFM backbone - the second-backbone arm of the RSI study.

    python src/xd_train_rtfm.py --variant E5 --seed 234 --run-name xd_rtfm_E5_s234
    python src/xd_train_rtfm.py --variant E3 --seed 234 --run-name xd_rtfm_E3_s234

Everything except the network is inherited from ``xd_train.py`` on purpose: same
``freq_text_options`` parser and variant table, same ``ClipVideoDataset`` loaders,
same 16x-frame evaluation, same pooled AUC/AP. The comparison this run exists to
support is **E3 - E5 inside RTFM**, not RTFM vs VadCLIP absolute numbers, so the
recipe is held fixed at the VadCLIP settings (10 epochs, lr 1e-5, batch 96,
``visual_length=256``, alpha annealing [0.1, 0.5]) rather than retuned. A retuned
RTFM would be a different experiment.

What is deliberately NOT inherited
----------------------------------
``freq_text_trainer.run_train`` is VadCLIP-shaped: it unpacks
``text_features, logits1, logits2 = model(...)``, adds ``CLASM`` on the text head
and ``prompt_dispersion`` on the text features. RTFM has no text branch, so the
loop is written out here with the pieces that still apply:

* ``F.binary_cross_entropy(topk_mean(sigmoid(logits)), is_anomaly)`` - bit-identical
  to ``CLAS2`` because the labels are already binary (``collate`` supplies
  ``item[4]``, so the one-hot round trip is unnecessary);
* RTFM's own feature-magnitude separation term, which is the mechanism RTFM is
  known for and would be dishonest to drop while claiming a faithful port;
* the same rank-1 injection as E1/E3, applied through the shared
  ``freq_text_aug.augment_sequence`` so the augmenter code path is literally the
  one the VadCLIP runs used.

Pre-registered risk (from ``experiments/00_backbone_audit.md``): RTFM's MIL pools
*by feature magnitude*, and the injection is magnitude-matched rank-1 noise on
low-score frames, so it necessarily perturbs RTFM's own selection statistic. That
is why N1 (matched iid noise) must be run in this backbone too - otherwise a gain
here cannot be attributed to structure. Run N1 with ``--variant N1``.
"""

import datetime
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import freq_text_options  # noqa: E402
from freq_text_aug import FreqTextAugmenter, augment_sequence  # noqa: E402
from freq_text_test import GT_FRAME_REPEAT  # noqa: E402,F401  (protocol constant, imported for readers)
from freq_text_text import build_class_directions  # noqa: E402
from freq_text_trainer import (  # noqa: E402
    _topk_mean_rows, build_loaders, build_test_loader, merge_batch, setup_seed)
from rtfm_eval import make_rtfm_test_fn  # noqa: E402
from rtfm_model import RTFM  # noqa: E402


def make_logger(run_dir, tag):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f'{tag}.log')

    def log(msg):
        line = f'{datetime.datetime.now():%H:%M:%S} {msg}'
        print(line, flush=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    return log


def build_augmenter(args, device, log):
    """Same construction block as ``freq_text_trainer.run_train``."""
    if args.aug_weight <= 0.0:
        return None, None
    augmenter = FreqTextAugmenter(
        dim=args.embed_dim, block_len=args.aug_block_len, mode=args.aug_mode,
        use_freq_gate=args.aug_freq_gate,
        alpha_range=(args.aug_alpha_lo, args.aug_alpha_hi),
        band_gain=args.aug_band_gain, gate_bands=args.aug_gate_bands,
        exclude_dc=args.aug_exclude_dc, delta_norm=args.aug_delta_norm,
        direction_source=args.aug_direction,
        gate_learnable=args.aug_gate_learnable,
        gate_random=args.aug_gate_random).to(device)
    dirs, dinfo = build_class_directions(args.clip_arch, args.dataset, device=device)
    log(f'augmenter: mode={args.aug_mode} direction={args.aug_direction} '
        f'alpha=[{args.aug_alpha_lo},{args.aug_alpha_hi}] block_len={args.aug_block_len} '
        f'| directions {dinfo if isinstance(dinfo, dict) else ""}')
    return augmenter, dirs


def main():
    parser = freq_text_options.build_parser('xd')
    parser.add_argument('--rtfm-mag-weight', type=float, default=1e-3,
                        help='RTFM feature-magnitude separation, positive term '
                             '(official: 0.001 on the abnormal top-k magnitude)')
    parser.add_argument('--rtfm-mag-neg', type=float, default=1e-4,
                        help='official 0.0001 on the normal top-k magnitude')
    parser.add_argument('--rtfm-dropout', type=float, default=0.7)
    args = freq_text_options.apply_variant(parser.parse_args())
    args = freq_text_options.resolve_paths(args, dataset='xd')

    tag = args.run_name or f'xd_rtfm_{args.variant}'
    run_dir = args.log_dir
    log = make_logger(run_dir, tag)
    setup_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'device={device} variant={args.variant} seed={args.seed} '
        f'aug_enabled={args.aug_weight > 0}')
    log('args: ' + ' | '.join(f'{k}={v}' for k, v in sorted(vars(args).items())))

    train_loader, anomaly_loader = build_loaders(args, 'xd', log)
    test_loader = build_test_loader(args, 'xd', log)
    test_fn = make_rtfm_test_fn('xd')

    model = RTFM(in_dim=args.embed_dim, drop_out=args.rtfm_dropout).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    log(f'RTFM params: {n_par/1e6:.2f} M (in_dim={args.embed_dim})')

    augmenter, dirs = build_augmenter(args, device, log)

    params = list(model.parameters())
    if augmenter is not None:
        params = params + list(augmenter.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.scheduler_milestones, gamma=args.scheduler_rate)

    gt = np.load(args.gt_path)
    gtseg = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabel = np.load(args.gt_label_path, allow_pickle=True)

    twin = anomaly_loader is not None
    history = []
    best, best_path, best_epoch = -1.0, None, 0

    for e in range(args.max_epoch):
        model.train()
        acc = {'mil': 0.0, 'mag': 0.0, 'aug': 0.0, 'n_aug': 0.0}
        n_batches = min(len(train_loader), len(anomaly_loader)) if twin else len(train_loader)
        if getattr(args, 'max_train_steps', 0):
            n_batches = min(n_batches, args.max_train_steps)
        normal_iter = iter(train_loader)
        anom_iter = iter(anomaly_loader) if twin else None
        t0 = time.time()

        for i in range(n_batches):
            items = [next(normal_iter)] + ([next(anom_iter)] if twin else [])
            feats, _labels, lengths, class_idx, binary = merge_batch(items)
            feats = feats.to(device)
            lengths = lengths.to(device)
            is_anom = binary.to(device).float().view(-1)

            logits, mfeat = model(feats, lengths)
            inst = _topk_mean_rows(torch.sigmoid(logits).reshape(logits.shape[0], -1), lengths)
            loss_mil = F.binary_cross_entropy(inst, is_anom)

            # RTFM's magnitude separation + the "normal bag should stay low" term.
            mag = _topk_mean_rows(mfeat.norm(dim=2), lengths)
            anom_mask = is_anom > 0.5
            nor_mask = ~anom_mask
            loss_mag = torch.zeros((), device=device)
            if bool(anom_mask.any()):
                loss_mag = loss_mag + args.rtfm_mag_weight * mag[anom_mask].mean()
            if bool(nor_mask.any()):
                loss_mag = (loss_mag
                            - args.rtfm_mag_neg * mag[nor_mask].mean()
                            + args.rtfm_mag_weight * ((1.0 - inst[nor_mask]) ** 2).mean())

            loss_aug = torch.zeros((), device=device)
            n_aug = 0.0
            if augmenter is not None and dirs is not None and bool(anom_mask.any()):
                f_sel = feats[anom_mask]
                len_sel = lengths[anom_mask]
                cls_sel = class_idx.to(device)[anom_mask].clamp(0, dirs.shape[0] - 1)
                with torch.no_grad():
                    ref_logits, _ = model(f_sel, len_sel)
                aug_feats, mask, _alpha = augment_sequence(
                    f_sel, dirs[cls_sel], augmenter, scores=ref_logits.squeeze(-1),
                    ratio=args.aug_ratio, lengths=len_sel)
                if bool(mask.any()):
                    aug_logits, _ = model(aug_feats, len_sel)
                    target = torch.full((int(mask.sum()),), float(args.aug_target),
                                        device=device)
                    loss_aug = args.aug_weight * F.binary_cross_entropy_with_logits(
                        aug_logits.squeeze(-1)[mask], target)
                    n_aug = float(mask.sum().item())

            loss = loss_mil + loss_mag + loss_aug
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(params, 10)
            optimizer.step()

            acc['mil'] += loss_mil.item()
            acc['mag'] += float(loss_mag.item())
            acc['aug'] += float(loss_aug.item())
            acc['n_aug'] += n_aug

            if args.log_every and (i + 1) % args.log_every == 0:
                k = i + 1
                log(f'[{tag}] ep {e+1}/{args.max_epoch} step {k}/{n_batches} '
                    f'mil {acc["mil"]/k:.4f} mag {acc["mag"]/k:.4f} '
                    f'aug {acc["aug"]/k:.4f} aug_frames/step {acc["n_aug"]/k:.1f}')
        scheduler.step()

        mean_aug = acc['n_aug'] / max(1, n_batches)
        log(f'[{tag}] epoch {e+1} done in {time.time()-t0:.0f}s | '
            f'mil {acc["mil"]/max(1,n_batches):.4f} '
            f'mag {acc["mag"]/max(1,n_batches):.4f} '
            f'aug {acc["aug"]/max(1,n_batches):.4f} | mean aug frames/step {mean_aug:.1f}')

        # RTFM has one head, so the selection metric is AP1 for XD (VadCLIP
        # selects on AP2 for XD; the head label is recorded in the history).
        _primary, extra = test_fn(model, test_loader, args.visual_length, None,
                                  gt, gtseg, gtlabel, device)
        metric = float(extra['AP1'])
        improved = metric > best
        history.append({'epoch': e + 1, 'AP1': metric,
                        'AUC1': float(extra['AUC1']),
                        'WV_AUC1': extra.get('WV_AUC1'),
                        'selection_head': 'AP1',
                        'loss_mil': acc['mil'] / max(1, n_batches),
                        'loss_mag': acc['mag'] / max(1, n_batches),
                        'loss_aug': acc['aug'] / max(1, n_batches),
                        'aug_frames_per_step': mean_aug,
                        'lr': float(optimizer.param_groups[0]['lr'])})
        log(f'[{tag}] ep {e+1}/{args.max_epoch} | {time.time()-t0:.0f}s | '
            f'AP1 {metric:.4f} AUC1 {extra["AUC1"]:.4f} | BEST AP1 {max(best, metric):.4f} | '
            f'lr {optimizer.param_groups[0]["lr"]:.2e}')

        if improved:
            best, best_epoch = metric, e + 1
            fname = (f'RTFM_auc{float(extra["AUC1"]):.4f}_ap{metric:.4f}_e{e+1}'
                     f'_variant={args.variant}_seed={args.seed}'
                     f'_lr={args.lr}_bs={args.batch_size}.pth')
            new_path = os.path.join(run_dir, fname)
            if best_path and os.path.exists(best_path):
                os.remove(best_path)          # one best checkpoint per run
            torch.save(model.state_dict(), new_path)
            best_path = new_path
            with open(os.path.join(run_dir, 'latest_best.txt'), 'w') as f:
                f.write(fname + '\n')
            log(f'[{tag}] new best AP1={metric:.4f} -> {fname}')

        with open(os.path.join(run_dir, f'{tag}_history.json'), 'w') as f:
            json.dump({'args': {k: v for k, v in vars(args).items()
                                if isinstance(v, (int, float, str, bool, list, type(None)))},
                       'backbone': 'RTFM', 'in_dim': args.embed_dim,
                       'n_params': n_par, 'selection_head': 'AP1',
                       'history': history}, f, indent=2)

    log(f'FINAL {tag} best AP1 = {best:.4f} (epoch {best_epoch}) checkpoint={best_path}')
    return best


if __name__ == '__main__':
    main()
