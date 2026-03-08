import torch.nn as nn

from clrnet.models.registry import NETS
from ..registry import build_aggregator, build_backbones, build_heads, build_necks


class FeatureRefineBlock(nn.Module):
    """Lightweight residual refinement block for FPN/backbone features."""

    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        out = self.bn(out)
        return self.act(out + x)


@NETS.register_module
class CLRErNet(nn.Module):
    """CLRNet variant with optional extra feature refinement stage.

    Config knobs (optional):
        clrernet = dict(
            enable_refine=True,
            refine_last_n=2,
        )
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbones(cfg)
        self.aggregator = build_aggregator(cfg) if cfg.haskey('aggregator') else None
        self.neck = build_necks(cfg) if cfg.haskey('neck') else None
        self.heads = build_heads(cfg)

        clrernet_cfg = cfg.clrernet if cfg.haskey('clrernet') else {}
        self.enable_refine = bool(clrernet_cfg.get('enable_refine', True))
        self.refine_last_n = int(clrernet_cfg.get('refine_last_n', 2))

        self.refine_blocks = None
        self.refine_start = 0

    def _init_refine_blocks(self, features):
        if not self.enable_refine:
            return
        if self.refine_blocks is not None:
            return
        if not isinstance(features, (list, tuple)):
            return
        if len(features) == 0:
            return

        n = max(1, min(self.refine_last_n, len(features)))
        self.refine_start = len(features) - n

        blocks = []
        for i in range(self.refine_start, len(features)):
            ch = int(features[i].shape[1])
            blocks.append(FeatureRefineBlock(ch))
        self.refine_blocks = nn.ModuleList(blocks)

    def _refine_features(self, features):
        if not self.enable_refine:
            return features
        if not isinstance(features, (list, tuple)):
            return features

        self._init_refine_blocks(features)
        if self.refine_blocks is None:
            return features

        out = list(features)
        for j, i in enumerate(range(self.refine_start, len(out))):
            out[i] = self.refine_blocks[j](out[i])
        return out

    def forward(self, batch):
        fea = self.backbone(batch['img'] if isinstance(batch, dict) else batch)

        if self.aggregator:
            fea[-1] = self.aggregator(fea[-1])

        if self.neck:
            fea = self.neck(fea)

        fea = self._refine_features(fea)

        if self.training:
            output = self.heads(fea, batch=batch)
        else:
            output = self.heads(fea)

        return output
