# deepswinlite/model.py
# -----------------------------------------------------------------------------
# DeepSwinLite (model-only): Swin backbone + MLFP + MSFA + AuxHead + light Decoder
# Dependencies: torch >= 1.10, timm >= 0.9
# -----------------------------------------------------------------------------

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required for DeepSwinLite backbone. Install with `pip install timm`."
    ) from e


# ---------------------------- MSFA module ----------------------------------- #

class MSFA(nn.Module):
    """
    Multi-Scale Feature Aggregation:
    parallel conv branches with different kernel sizes (1,3,5,7),
    then channel concat + 1x1 fuse.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b1 = ConvBNAct(in_ch, out_ch // 2, k=1, act='relu')  # lightweight linear proj
        # ensure total <= out_ch after concat, adjust splits
        remain = out_ch - (out_ch // 2)
        part = max(1, remain // 3)
        last = remain - 2 * part

        self.b3 = ConvBNAct(in_ch, part, k=3, act='relu')
        self.b5 = ConvBNAct(in_ch, part, k=5, act='relu')
        self.b7 = ConvBNAct(in_ch, last, k=7, act='relu')

        self.fuse = ConvBNAct(out_ch, out_ch, k=1, act='relu')

    def forward(self, x):
        x1 = self.b1(x)
        x3 = self.b3(x)
        x5 = self.b5(x)
        x7 = self.b7(x)
        x  = torch.cat([x1, x3, x5, x7], dim=1)
        return self.fuse(x)


# ---------------------------- MLFP module ----------------------------------- #

class MLFP(nn.Module):
    """
    Multi-Level Feature Pyramid:
    - lateral 1x1 to align channels
    - top-down fusion with upsample+add
    - context via dilated convs
    """
    def __init__(self, in_channels: Tuple[int, int, int, int], out_ch: int = 128,
                 dilations=(1, 2, 4)):
        super().__init__()
        c1, c2, c3, c4 = in_channels  # from backbone stages (low->high)
        self.l4 = ConvBNAct(c4, out_ch, k=1, act='relu')
        self.l3 = ConvBNAct(c3, out_ch, k=1, act='relu')
        self.l2 = ConvBNAct(c2, out_ch, k=1, act='relu')
        self.l1 = ConvBNAct(c1, out_ch, k=1, act='relu')

        # post-fusion smoothing
        self.s4 = ConvBNAct(out_ch, out_ch, k=3, act='relu')
        self.s3 = ConvBNAct(out_ch, out_ch, k=3, act='relu')
        self.s2 = ConvBNAct(out_ch, out_ch, k=3, act='relu')
        self.s1 = ConvBNAct(out_ch, out_ch, k=3, act='relu')

        # dilated context on merged map
        self.ctx = nn.ModuleList([ConvBNAct(out_ch, out_ch, k=3, d=d, act='relu')
                                  for d in dilations])
        self.fuse = ConvBNAct(out_ch * (len(dilations) + 1), out_ch, k=1, act='relu')

    @staticmethod
    def _upsample_add(x, y):
        return F.interpolate(x, size=y.shape[-2:], mode='bilinear', align_corners=False) + y

    def forward(self, feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        f1, f2, f3, f4 = feats  # resolutions: 1/4, 1/8, 1/16, 1/32 of input (typical for Swin)
        p4 = self.s4(self.l4(f4))
        p3 = self.s3(self._upsample_add(p4, self.l3(f3)))
        p2 = self.s2(self._upsample_add(p3, self.l2(f2)))
        p1 = self.s1(self._upsample_add(p2, self.l1(f1)))  # highest resolution pyramid map

        # context on p1
        xs = [p1] + [m(p1) for m in self.ctx]
        x = torch.cat(xs, dim=1)
        x = self.fuse(x)
        return x  # pyramid head at ~1/4 input resolution (given Swin features_only stride=4 for f1)


# ------------------------------ AuxHead ------------------------------------- #

class AuxHead(nn.Module):
    """
    Auxiliary head fed from an intermediate feature map.
    Includes two conv blocks + SE + dropout + 1x1 classifier.
    """
    def __init__(self, in_ch, num_classes=2, dropout=0.2):
        super().__init__()
        self.block1 = nn.Sequential(
            ConvBNAct(in_ch, in_ch // 2, k=3, act='lrelu'),
            ConvBNAct(in_ch // 2, in_ch // 2, k=3, act='silu'),
        )
        self.block2 = ConvBNAct(in_ch // 2, in_ch // 4, k=3, act='relu')
        self.se     = SEBlock(in_ch // 4, r=8)
        self.drop   = nn.Dropout2d(p=dropout)
        self.cls    = nn.Conv2d(in_ch // 4, num_classes, kernel_size=1, bias=True)

    def forward(self, x, out_size: Tuple[int, int]):
        x = self.block1(x)
        x = self.block2(x)
        x = self.se(x)
        x = self.drop(x)
        x = self.cls(x)
        # upsample aux logits to input size for convenience
        x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        return x


# ------------------------------ Decoder ------------------------------------- #

class LightDecoder(nn.Module):
    """
    Lightweight decoder that upsamples pyramid feature (≈1/4 res) back to input resolution.
    """
    def __init__(self, in_ch=128, mid_ch=64, out_ch=32, num_classes=2):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_ch, mid_ch, kernel_size=2, stride=2)  # 1/4 -> 1/2
        self.c1  = ConvBNAct(mid_ch, mid_ch, k=3, act='relu')
        self.up2 = nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=2, stride=2)  # 1/2 -> 1/1
        self.c2  = ConvBNAct(out_ch, out_ch, k=3, act='relu')
        self.head = nn.Conv2d(out_ch, num_classes, kernel_size=1, bias=True)

    def forward(self, x, out_size: Tuple[int, int]):
        x = self.up1(x)
        x = self.c1(x)
        x = self.up2(x)
        x = self.c2(x)
        x = self.head(x)
        # ensure exact size match (handles odd dimensions)
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        return x


# ---------------------------- Backbone wrapper ------------------------------ #

def create_swin_backbone(name: str = "swin_tiny_patch4_window7_224",
                         pretrained: bool = False):
    """
    Returns a timm backbone with features_only=True (out_indices=(0,1,2,3)).
    Common channel configs for swin_tiny: [96, 192, 384, 768]
    """
    backbone = timm.create_model(
        name,
        features_only=True,
        pretrained=pretrained,
        out_indices=(0, 1, 2, 3),
    )
    feat_info = backbone.feature_info
    channels = tuple(fi["num_chs"] for fi in feat_info)
    return backbone, channels


# ------------------------------- Main model --------------------------------- #

class DeepSwinLite(nn.Module):
    """
    Model-only implementation:
      - Swin backbone (timm)
      - MLFP pyramid head
      - MSFA enrichment
      - Light decoder to full-res logits
      - Optional AuxHead
    """
    def __init__(
        self,
        num_classes: int = 2,
        backbone_name: str = "swin_tiny_patch4_window7_224",
        backbone_pretrained: bool = False,
        pyramid_ch: int = 128,
        aux_head: bool = True,
        aux_from_stage: int = 2,  # take aux from feature stage i (0..3), default stage-3 (index 2)
        dropout_aux: float = 0.2,
    ):
        super().__init__()

        self.backbone, in_chs = create_swin_backbone(backbone_name, backbone_pretrained)
        # in_chs ~ (96, 192, 384, 768) for swin_tiny
        self.mlfp = MLFP(in_channels=in_chs, out_ch=pyramid_ch, dilations=(1, 2, 4))
        self.msfa = MSFA(in_ch=pyramid_ch, out_ch=pyramid_ch)
        self.dec  = LightDecoder(in_ch=pyramid_ch, mid_ch=64, out_ch=32, num_classes=num_classes)

        self.use_aux = aux_head
        self.aux_from_stage = int(aux_from_stage)
        if self.use_aux:
            in_aux = in_chs[self.aux_from_stage]
            self.aux = AuxHead(in_ch=in_aux, num_classes=num_classes, dropout=dropout_aux)

        # initialize final conv layers with small std for stability
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B,3,H,W)
        Returns:
            dict with:
              - 'logits': (B,num_classes,H,W)
              - 'aux': (B,num_classes,H,W) if aux_head=True
        """
        b, c, h, w = x.shape

        # backbone features: list of 4 tensors (res strides typically 4,8,16,32)
        feats = self.backbone(x)  # [f1,f2,f3,f4]
        # main path
        p   = self.mlfp(feats)            # ~ 1/4 resolution
        p   = self.msfa(p)                # enrich local+global context
        out = self.dec(p, out_size=(h, w))

        out_dict: Dict[str, torch.Tensor] = {"logits": out}

        # auxiliary supervision path (upsampled to input size)
        if self.use_aux:
            aux_feat = feats[self.aux_from_stage]
            aux_out  = self.aux(aux_feat, out_size=(h, w))
            out_dict["aux"] = aux_out

        return out_dict


# ------------------------------ quick self-test ----------------------------- #
if __name__ == "__main__":
    model = DeepSwinLite(num_classes=2, backbone_pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print("logits:", y["logits"].shape)
    if "aux" in y:
        print("aux   :", y["aux"].shape)

