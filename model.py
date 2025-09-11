"""
model.py — DeepSwinLite (Swin backbone + MLFP + MSFA + Light Decoder + Aux Head)

- Backbone: Swin Transformer via timm (features_only=True, 4-stage features)
- MLFP: Multi-Level Feature Pyramid (lateral 1x1 + top-down upsample/add + smoothing)
- MSFA: Multi-Scale Feature Aggregation (parallel branches with 1/3/5/7 kernels)
- Decoder: Lightweight two-stage up-conv to full-resolution logits
- AuxHead: Auxiliary output from MSFA map (upsampled to input size)

Outputs:
  main_logits, aux_logits  # (B, num_classes, H, W)

Requirements:
  pip install torch timm
"""


from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ----------------------------- Aux Head ------------------------------------- #

class AuxHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(in_channels // 2)
        self.act1  = nn.LeakyReLU(0.1, inplace=True)

        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1, dilation=2, bias=False)
        self.bn2   = nn.BatchNorm2d(in_channels // 4)
        self.act2  = nn.SiLU(inplace=True)

   
        mid = max(8, in_channels // 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels // 4, mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels // 4, kernel_size=1),
            nn.Sigmoid(),
        )

        self.drop = nn.Dropout2d(dropout_rate)
        self.cls  = nn.Conv2d(in_channels // 4, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = x * self.se(x)
        x = self.drop(x)
        x = self.cls(x)
        return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)


# ------------------------------ MLFP ---------------------------------------- #

class MLFP(nn.Module):
    """
    Multi-Level Feature Pyramid:
    lateral 1x1 -> top-down upsample+add -> 3x3 smoothing
    """
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list])
        self.smooth  = nn.ModuleList([nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                                      for _ in in_channels_list])

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
     
        c1, c2, c3, c4 = feats
        p4 = self.lateral[3](c4)
        p3 = self.lateral[2](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lateral[1](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p1 = self.lateral[0](c1) + F.interpolate(p2, size=c1.shape[-2:], mode="nearest")

        # smoothing
        p4 = self.smooth[3](p4)
        p3 = self.smooth[2](p3)
        p2 = self.smooth[1](p2)
        p1 = self.smooth[0](p1)
        return p1 


# ------------------------------- MSFA --------------------------------------- #

class MSFA(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.fuse  = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        return self.fuse(torch.cat([x1, x3, x5, x7], dim=1))


# ---------------------------- Light Decoder --------------------------------- #

class LightDecoder(nn.Module):

    def __init__(self, in_ch: int = 128, mid_ch: int = 64, out_ch: int = 32, num_classes: int = 2):
        super().__init__()
        self.up1   = nn.ConvTranspose2d(in_ch, mid_ch, kernel_size=2, stride=2)  
        self.block = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.up2  = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2)   
        self.head = nn.Conv2d(out_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        x = self.up1(x)
        x = self.block(x)
        x = self.up2(x)
        x = self.head(x)
        if x.shape[-2:] != out_hw:
            x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return x


# ------------------------------- Main --------------------------------------- #

class DeepSwinLite(nn.Module):

    def __init__(
        self,
        num_classes: int = 2,
        backbone_name: str = "swin_tiny_patch4_window7_224",
        backbone_pretrained: bool = True,
        mlfp_out_channels: int = 128,
        apply_softmax: bool = False,
    ):
        super().__init__()

       
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=backbone_pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        chs = [fi["num_chs"] for fi in self.backbone.feature_info]


        self.mlfp = MLFP(chs, mlfp_out_channels)
        self.msfa = MSFA(mlfp_out_channels, mlfp_out_channels)


        self.decoder  = LightDecoder(in_ch=mlfp_out_channels, mid_ch=64, out_ch=32, num_classes=num_classes)
        self.aux_head = AuxHead(in_channels=mlfp_out_channels, num_classes=num_classes, dropout_rate=0.3)

        self.apply_softmax = apply_softmax


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor):

        b, c, h, w = x.shape

        feats = self.backbone(x)  
        p = self.mlfp(feats)    
        p = self.msfa(p)

        aux = self.aux_head(p, (h, w))
        main = self.decoder(p, (h, w))
        if self.apply_softmax:
            main = torch.softmax(main, dim=1)
        return main, aux


# ----------------------------- Quick Test ----------------------------------- #

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSwinLite(num_classes=2, backbone_pretrained=False, apply_softmax=False).to(device)
    x = torch.randn(1, 3, 512, 512, device=device)
    main, aux = model(x)
    print("Main:", main.shape)  
    print("Aux :", aux.shape)   

