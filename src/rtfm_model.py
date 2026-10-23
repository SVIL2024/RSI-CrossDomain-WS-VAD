"""RTFM as a SECOND backbone for the RSI study (CLIP-feature variant).

Why this file exists
--------------------
The submission's reviewers asked for a second backbone, which is a normal request
for a paper whose claim is about a *mechanism* rather than a score. MGFN was
considered and rejected (see ``experiments/00_backbone_audit.md``): its inductive
bias is "feature magnitude == abnormality prior", and this project measured
``||f - mu||`` on UCF-CLIP to score **0.4520 pooled AUC, below chance**, so it
would fight the injection instead of hosting it.

RTFM is the right second backbone: no text branch, no CLIP backbone, no graph - a
pure temporal MIL detector, so a gain that survives here is not a CLIP artefact.

What is copied from the official implementation and what is not
---------------------------------------------------------------
Copied faithfully from ``RTFM/model.py`` (Apache-2.0, Tian et al., CVPR 2021):

* ``weight_init`` (xavier + zero bias),
* ``Aggregate`` - three dilated Conv1d branches (dilation 1/2/4) concatenated with
  a non-local block, then a 1x1 fusion conv and a residual,
* ``NonLocalBlock1D`` with ``sub_sample=False``, ``bn_layer=True``, zero-init ``W``,
* the head ``Linear(2048,512) -> ReLU -> Dropout -> Linear(512,128) -> ReLU ->
  Dropout -> Linear(128,1)`` with dropout 0.7.

NOT copied: the official ``Model.forward``. That function hard-codes the I3D
plumbing - a bag of ``batch_size*10`` normal rows followed by abnormal ones, 10
crops per video, ``num_segments=32``, and top-k selection by *feature magnitude*
with ``k = 32//10``. Feeding it our loader (one video per row, real frame counts,
right-padded to ``visual_length``) would silently mis-slice the batch, which is
exactly the failure mode that made the old ``E:\\program\\RTFM`` run useless. So the
tensor plumbing is native here and only the *network* is borrowed.

Interface difference, stated for the record
-------------------------------------------
``forward(x, lengths)`` takes ``(B, T, C)`` CLIP features and returns
``(logits (B,T,1), feats (B,T,2048))``. ``lengths`` is accepted for signature
symmetry with ``CLIPVAD`` but is not needed: the padding mask is applied by the
loss, not by the network (the official code also ignores padding inside
``Aggregate`` and relies on the top-k pooling).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init

__all__ = ['RTFM', 'weight_init']


def weight_init(m):
    """Official RTFM initialisation: xavier on conv/linear, zero bias."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        if m.weight is not None:
            torch_init.xavier_uniform_(m.weight)
        if getattr(m, 'bias', None) is not None:
            m.bias.data.fill_(0)


class NonLocalBlock1D(nn.Module):
    """1-D non-local block, the only specialisation RTFM uses (``dimension=1``).

    The official file implements a generic ``_NonLocalBlockND`` for 1/2/3-D and
    subclasses it; only the 1-D path is ever instantiated, so it is written out
    directly here instead of carrying three dead branches.

    ``sub_sample=False`` and ``bn_layer=True`` as in ``NONLocalBlock1D(512,
    sub_sample=False, bn_layer=True)``: no max-pool on ``g``, and ``W`` ends in a
    BatchNorm whose weights are zero-initialised, so the block starts as an
    identity mapping.
    """

    def __init__(self, in_channels, inter_channels=None, sub_sample=False, bn_layer=True):
        super().__init__()
        # RTFM only ever instantiates ``sub_sample=False, bn_layer=True``; the
        # keywords are kept so the call site reads like the official code, and any
        # other combination fails loudly instead of silently changing the block.
        if sub_sample or not bn_layer:
            raise NotImplementedError(
                'this port implements only RTFM\'s sub_sample=False/bn_layer=True '
                f'non-local block (got sub_sample={sub_sample}, bn_layer={bn_layer})')
        inter_channels = inter_channels or max(1, in_channels // 2)
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv1d(in_channels, inter_channels, kernel_size=1)
        self.W = nn.Sequential(
            nn.Conv1d(inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm1d(in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.theta = nn.Conv1d(in_channels, inter_channels, kernel_size=1)
        self.phi = nn.Conv1d(in_channels, inter_channels, kernel_size=1)

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)                       # (B, T, C/2)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)               # (B, T, C/2)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)   # (B, C/2, T)

        f = torch.matmul(theta_x, phi_x)                 # (B, T, T)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)                   # (B, T, C/2)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        return self.W(y) + x


class Aggregate(nn.Module):
    """RTFM's temporal encoder: three dilations + non-local, fused, plus residual.

    The official ``Aggregate`` hard-codes ``in_channels=2048`` in ``conv_4`` and
    ``conv_5`` because the residual stream is 2048-wide. Two edits, both required
    to accept 512-d CLIP features and both shape-preserving at ``len_feature=2048``:

    * ``conv_4`` reads ``len_feature`` (it consumes the *input*, not ``out_d``),
    * the residual is projected when ``len_feature != 2048``; at 2048 the
      projection is omitted entirely so the official parameter set is unchanged.
    """

    def __init__(self, len_feature=512):
        super().__init__()
        self.len_feature = len_feature

        def branch(dilation):
            return nn.Sequential(
                nn.Conv1d(len_feature, 512, kernel_size=3, stride=1,
                          dilation=dilation, padding=dilation),
                nn.ReLU(),
                nn.BatchNorm1d(512),
            )

        self.conv_1 = branch(1)
        self.conv_2 = branch(2)
        self.conv_3 = branch(4)
        # 3 x 512 dilated branches + 512 non-local = 2048 channels
        self.conv_4 = nn.Sequential(
            nn.Conv1d(len_feature, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(2048, 2048, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
        )
        self.non_local = NonLocalBlock1D(512, sub_sample=False, bn_layer=True)

        self.residual_proj = (None if len_feature == 2048
                              else nn.Conv1d(len_feature, 2048, kernel_size=1,
                                             bias=False))

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        out = x.permute(0, 2, 1)
        residual = out

        out1 = self.conv_1(out)
        out2 = self.conv_2(out)
        out3 = self.conv_3(out)
        out_d = torch.cat((out1, out2, out3), dim=1)     # (B, 1536, T)

        out = self.conv_4(out)                           # (B, 512, T)
        out = self.non_local(out)
        out = torch.cat((out_d, out), dim=1)             # (B, 2048, T)
        out = self.conv_5(out)                           # (B, 2048, T)

        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        out = out + residual
        return out.permute(0, 2, 1)                      # (B, T, 2048)


class RTFM(nn.Module):
    """RTFM network for CLIP features, frame-level score interface.

    ``in_dim`` is the feature width (512 for CLIP ViT-B/16, 2048 for I3D). The
    head width is fixed at 2048 because that is ``Aggregate``'s output width,
    independent of ``in_dim``.
    """

    def __init__(self, in_dim=512, drop_out=0.7):
        super().__init__()
        self.in_dim = in_dim
        self.aggregate = Aggregate(len_feature=in_dim)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.drop_out = nn.Dropout(drop_out)
        self.relu = nn.ReLU()
        self.apply(weight_init)

    def forward(self, x, lengths=None):
        """x: ``(B, T, in_dim)`` -> ``(logits (B,T,1), feats (B,T,2048))``.

        Raw logits, deliberately: ``freq_text_trainer.CLAS2`` takes logits and
        applies ``sigmoid`` + top-k itself, so feeding it RTFM's logits makes the
        MIL objective **bit-identical** to VadCLIP's ``loss1``. That is the point
        of the exercise - the backbone is the only thing that changes.
        """
        feats = self.aggregate(x)
        feats = self.drop_out(feats)

        scores = self.relu(self.fc1(feats))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        logits = self.fc3(scores)                        # (B, T, 1)
        return logits, feats
