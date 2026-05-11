import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Union
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image.psnr import PeakSignalNoiseRatio as PSNR


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    return arr.unsqueeze(0)


def _to_tensor(images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(images, Image.Image):
        return _pil_to_tensor(images)
    return torch.cat([_pil_to_tensor(img) for img in images], dim=0)


class ReferenceMetrics:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self._ssim = SSIM(data_range=1.0).to(self.device)
        self._lpips = LPIPS(net_type="alex").to(self.device)
        self._psnr = PSNR(data_range=1.0).to(self.device)

    def _prep(self, frames: List[Image.Image], ref: Image.Image):
        f = _to_tensor(frames).to(self.device)
        r = _pil_to_tensor(ref).to(self.device)
        if f.shape[2:] != r.shape[2:]:
            r = F.interpolate(r, size=f.shape[2:], mode="bilinear", align_corners=False)
        return f, r.expand_as(f)

    def ssim(self, frames: List[Image.Image], ref: Image.Image) -> float:
        f, r = self._prep(frames, ref)
        return self._ssim(f, r).item()

    def lpips(self, frames: List[Image.Image], ref: Image.Image) -> float:
        f, r = self._prep(frames, ref)
        return self._lpips(f, r).mean().item()

    def psnr(self, frames: List[Image.Image], ref: Image.Image) -> float:
        f, r = self._prep(frames, ref)
        return self._psnr(f, r).item()

    def temporal_lpips(self, frames: List[Image.Image]) -> float:
        if len(frames) < 2:
            return 0.0
        scores = []
        for i in range(len(frames) - 1):
            a = _pil_to_tensor(frames[i]).to(self.device)
            b = _pil_to_tensor(frames[i + 1]).to(self.device)
            scores.append(self._lpips(a, b).item())
        return sum(scores) / len(scores)

    def all(self, frames: List[Image.Image], ref: Image.Image) -> dict:
        return {
            "ssim": self.ssim(frames, ref),
            "lpips": self.lpips(frames, ref),
            "psnr": self.psnr(frames, ref),
            "temporal_lpips": self.temporal_lpips(frames),
        }
