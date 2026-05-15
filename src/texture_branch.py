"""Texture enhancement branch for pixel-level uncertainty-aware VSR.

The module keeps the original heuristic enhancements for backward compatibility,
and adds a Real-ESRGAN backend that can be used as the texture branch in the
uncertainty-aware pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.parse import urlparse
from urllib.request import urlretrieve

import cv2
import numpy as np

try:  # pragma: no cover - optional dependency
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from realesrgan import RealESRGANer
except Exception:  # pragma: no cover - optional dependency
    RRDBNet = None
    RealESRGANer = None
    load_file_from_url = None


_DEFAULT_WEIGHT_URLS: dict[str, str] = {
    "RealESRGAN_x4plus": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
        "RealESRGAN_x4plus.pth"
    ),
    "RealESRGAN_x4plus_anime_6B": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/"
        "RealESRGAN_x4plus_anime_6B.pth"
    ),
    "RealESRGAN_x2plus": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
        "RealESRGAN_x2plus.pth"
    ),
}


def _normalize_kernel_size(size: int) -> int:
    if size < 3 or size % 2 == 0:
        return max(3, size | 1)
    return size


def _ensure_bgr_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unsupported frame shape: {frame.shape}")


def _cache_dir() -> Path:
    return Path.home() / ".cache" / "roi_vsr_project" / "realesrgan"


def _download_weight(url: str, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path
    if load_file_from_url is not None:
        try:
            return Path(load_file_from_url(url, model_dir=str(target_path.parent)))
        except Exception:
            pass
    urlretrieve(url, target_path)
    return target_path


def _resolve_weight_path(model_name: str, model_path: str | Path | None) -> Path:
    if model_path is not None:
        path = Path(model_path)
        if path.is_file():
            return path
        if str(model_path).startswith(("http://", "https://")):
            file_name = Path(urlparse(str(model_path)).path).name or f"{model_name}.pth"
            return _download_weight(str(model_path), _cache_dir() / file_name)
        raise FileNotFoundError(f"Real-ESRGAN weight not found: {path}")

    default_url = _DEFAULT_WEIGHT_URLS.get(model_name)
    if default_url is None:
        raise ValueError(f"No default weight URL registered for model: {model_name}")

    file_name = Path(urlparse(default_url).path).name
    return _download_weight(default_url, _cache_dir() / file_name)


def _build_rrdbnet(model_name: str, scale: int) -> Any:
    if RRDBNet is None:
        raise ImportError(
            "Real-ESRGAN dependencies are not installed. Install `realesrgan` "
            "and `basicsr` in the BasicVSR++ environment."
        )

    if model_name == "RealESRGAN_x4plus":
        return RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale,
        )
    if model_name == "RealESRGAN_x4plus_anime_6B":
        return RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=6,
            num_grow_ch=32,
            scale=scale,
        )
    if model_name == "RealESRGAN_x2plus":
        return RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale,
        )
    raise ValueError(f"Unsupported Real-ESRGAN model: {model_name}")


@dataclass
class RealESRGANTextureEnhancer:
    """Lazy-loaded Real-ESRGAN texture enhancement backend."""

    model_name: str = "RealESRGAN_x4plus"
    scale: int = 4
    model_path: str | Path | None = None
    tile: int = 0
    tile_pad: int = 10
    pre_pad: int = 0
    half: bool = True
    gpu_id: int = 0
    dni_weight: Any = None

    _enhancer: Any = field(default=None, init=False, repr=False)

    def _build(self) -> Any:
        if RealESRGANer is None:
            raise ImportError(
                "Real-ESRGAN is not installed. Run `pip install realesrgan basicsr` "
                "inside the BasicVSR++ environment."
            )

        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except Exception:  # pragma: no cover - torch always available in target env
            cuda_available = False

        model = _build_rrdbnet(self.model_name, self.scale)
        weight_path = _resolve_weight_path(self.model_name, self.model_path)
        device = torch.device(f"cuda:{self.gpu_id}") if cuda_available else torch.device("cpu")
        return RealESRGANer(
            scale=self.scale,
            model_path=str(weight_path),
            model=model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=bool(self.half and cuda_available),
            device=device,
            gpu_id=self.gpu_id if cuda_available else None,
            dni_weight=self.dni_weight,
        )

    @property
    def enhancer(self) -> Any:
        if self._enhancer is None:
            self._enhancer = self._build()
        return self._enhancer

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        frame_bgr = _ensure_bgr_frame(frame)
        output, _ = self.enhancer.enhance(frame_bgr, outscale=self.scale)
        return output

    def enhance_sequence(self, frames: Sequence[np.ndarray]) -> list[np.ndarray]:
        return [self.enhance_frame(frame) for frame in frames]


def apply_unsharp_mask(
    frame: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
    amount: float = 1.0,
) -> np.ndarray:
    """Apply unsharp masking to enhance edges and details."""
    kernel_size = _normalize_kernel_size(kernel_size)
    frame_f = frame.astype(np.float32)
    blurred = cv2.GaussianBlur(frame_f, (kernel_size, kernel_size), sigma)
    sharpened = frame_f + amount * (frame_f - blurred)
    return np.clip(sharpened, 0, 255).astype(frame.dtype)


def apply_local_contrast_enhancement(
    frame: np.ndarray,
    window_size: int = 31,
    strength: float = 1.2,
) -> np.ndarray:
    """Apply local contrast enhancement (CLAHE-like) to boost local details."""
    window_size = _normalize_kernel_size(window_size)

    if frame.ndim == 3 and frame.shape[2] == 3:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[..., 0].astype(np.float32)

        local_mean = cv2.blur(l_channel, (window_size, window_size))
        local_sq_mean = cv2.blur(l_channel ** 2, (window_size, window_size))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

        with np.errstate(divide="ignore", invalid="ignore"):
            enhanced_l = np.where(
                local_std > 0,
                (l_channel - local_mean) * strength + local_mean,
                l_channel,
            )

        lab[..., 0] = np.clip(enhanced_l, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = frame.astype(np.float32)
    local_mean = cv2.blur(gray, (window_size, window_size))
    local_sq_mean = cv2.blur(gray ** 2, (window_size, window_size))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

    with np.errstate(divide="ignore", invalid="ignore"):
        enhanced = np.where(
            local_std > 0,
            (gray - local_mean) * strength + local_mean,
            gray,
        )
    return np.clip(enhanced, 0, 255).astype(frame.dtype)


def enhance_texture_frame(
    frame: np.ndarray,
    method: str = "realesrgan",
    sharpen_amount: float = 1.0,
    blur_ksize: int = 5,
    realesrgan_enhancer: RealESRGANTextureEnhancer | None = None,
    realesrgan_config: Optional[dict[str, Any]] = None,
    gpu_id: int = 0,
) -> np.ndarray:
    """Enhance texture/details of a single frame."""
    normalized_method = method.lower().strip()

    if normalized_method == "realesrgan":
        enhancer = realesrgan_enhancer
        if enhancer is None:
            config = realesrgan_config or {}
            enhancer = RealESRGANTextureEnhancer(
                model_name=str(config.get("model_name", "RealESRGAN_x4plus")),
                scale=int(config.get("scale", 4)),
                model_path=config.get("model_path"),
                tile=int(config.get("tile", 0)),
                tile_pad=int(config.get("tile_pad", 10)),
                pre_pad=int(config.get("pre_pad", 0)),
                half=bool(config.get("half", True)),
                gpu_id=gpu_id,
                dni_weight=config.get("dni_weight"),
            )
        return enhancer.enhance_frame(frame)

    if normalized_method == "unsharp":
        return apply_unsharp_mask(
            frame,
            kernel_size=blur_ksize,
            sigma=1.0,
            amount=sharpen_amount,
        )
    if normalized_method == "local_contrast":
        return apply_local_contrast_enhancement(
            frame,
            window_size=max(3, blur_ksize | 1),
            strength=1.2,
        )
    if normalized_method == "hybrid":
        sharpened = apply_unsharp_mask(
            frame,
            kernel_size=blur_ksize,
            amount=sharpen_amount * 0.6,
        )
        return apply_local_contrast_enhancement(sharpened, strength=1.1)

    raise ValueError(f"Unknown enhancement method: {method}")


def enhance_texture_sequence(
    frames: Sequence[np.ndarray],
    method: str = "realesrgan",
    sharpen_amount: float = 1.0,
    blur_ksize: int = 5,
    realesrgan_config: Optional[dict[str, Any]] = None,
    gpu_id: int = 0,
) -> list[np.ndarray]:
    """Enhance texture of a frame sequence."""
    normalized_method = method.lower().strip()

    if normalized_method == "realesrgan":
        config = realesrgan_config or {}
        enhancer = RealESRGANTextureEnhancer(
            model_name=str(config.get("model_name", "RealESRGAN_x4plus")),
            scale=int(config.get("scale", 4)),
            model_path=config.get("model_path"),
            tile=int(config.get("tile", 0)),
            tile_pad=int(config.get("tile_pad", 10)),
            pre_pad=int(config.get("pre_pad", 0)),
            half=bool(config.get("half", True)),
            gpu_id=gpu_id,
            dni_weight=config.get("dni_weight"),
        )
        return enhancer.enhance_sequence(frames)

    return [
        enhance_texture_frame(
            frame,
            method=normalized_method,
            sharpen_amount=sharpen_amount,
            blur_ksize=blur_ksize,
            gpu_id=gpu_id,
        )
        for frame in frames
    ]


def placeholder_generative_enhance(
    frame: np.ndarray,
    realesrgan_config: Optional[dict[str, Any]] = None,
    gpu_id: int = 0,
) -> np.ndarray:
    """Compatibility wrapper that now uses Real-ESRGAN by default."""
    return enhance_texture_frame(
        frame,
        method="realesrgan",
        realesrgan_config=realesrgan_config,
        gpu_id=gpu_id,
    )
