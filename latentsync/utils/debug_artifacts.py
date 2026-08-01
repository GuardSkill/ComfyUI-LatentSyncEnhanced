"""Opt-in, bounded image dumps for tracing face restoration artifacts."""

import os
from pathlib import Path
from typing import Any


DEBUG_ENV_VAR = "LATENTSYNC_DEBUG_ARTIFACTS"
# Keep dumps beside the custom node so they are easy to find and remove.
DEBUG_DIR = Path(__file__).resolve().parents[2] / "debug_artifacts"
MAX_DEBUG_FRAMES = 10


def artifact_debug_enabled() -> bool:
    """Return True only when artifact image dumping is explicitly enabled."""

    return os.getenv(DEBUG_ENV_VAR) == "1"


def save_artifact_image(
    name: str,
    frame_index: int,
    image: Any,
    value_range: str,
    color_order: str = "rgb",
) -> bool:
    """Save one RGB/BGR/gray debug image when enabled and within the limit.

    ``value_range`` must be one of ``minus_one_one``, ``zero_one``, or
    ``zero_255``. ``color_order`` describes three-channel input and must be
    ``rgb`` or ``bgr``. Torch tensors are detached and copied to CPU only after
    the environment-variable and frame-limit guards pass.
    """

    if not artifact_debug_enabled() or not 0 <= frame_index < MAX_DEBUG_FRAMES:
        return False

    import cv2
    import numpy as np

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)

    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"Expected a single debug image, got shape: {array.shape}")
        array = array[0]

    # Prefer HWC when the final dimension is an image channel count. This
    # avoids misclassifying a valid HWC image whose height happens to be 3/4.
    if (
        array.ndim == 3
        and array.shape[-1] not in (1, 3, 4)
        and array.shape[0] in (1, 3, 4)
    ):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]

    array = array.astype(np.float32, copy=False)
    if value_range == "minus_one_one":
        array = (array / 2.0 + 0.5) * 255.0
    elif value_range == "zero_one":
        array = array * 255.0
    elif value_range != "zero_255":
        raise ValueError(f"Unsupported debug image value_range: {value_range}")

    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    array = np.clip(array, 0.0, 255.0).round().astype(np.uint8)

    if color_order not in ("rgb", "bgr"):
        raise ValueError(f"Unsupported debug image color_order: {color_order}")

    # OpenCV expects BGR when writing color images.
    if array.ndim == 3 and array.shape[-1] == 3:
        if color_order == "rgb":
            array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    elif array.ndim not in (2, 3):
        raise ValueError(f"Unsupported debug image shape: {array.shape}")

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DEBUG_DIR / f"{frame_index:06d}_{name}.png"
    if not cv2.imwrite(str(output_path), array):
        raise OSError(f"Failed to write debug artifact: {output_path}")
    return True
