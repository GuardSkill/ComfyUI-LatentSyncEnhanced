"""Yaw-aware canonical mask adaptation and facial contour clipping."""

import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F


YAW_THRESHOLD_ENV = "LATENTSYNC_MOUTH_YAW_THRESHOLD"
MAX_SHRINK_ENV = "LATENTSYNC_MOUTH_MAX_HORIZONTAL_SHRINK"
MAX_SHIFT_ENV = "LATENTSYNC_MOUTH_MAX_HORIZONTAL_SHIFT"
CONTOUR_FEATHER_ENV = "LATENTSYNC_MOUTH_CONTOUR_FEATHER"
MAX_EDITABLE_COVERAGE_ENV = "LATENTSYNC_MOUTH_MAX_EDITABLE_COVERAGE"

FACE_CONTOUR_INDICES = [
    1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0,
    24, 23, 22, 21, 20, 19, 18, 32, 31, 30, 29, 28, 27, 26, 25, 17,
]


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, default))
    except (TypeError, ValueError):
        warnings.warn(f"Invalid {name}; using default {default}", RuntimeWarning)
        value = default
    if not np.isfinite(value) or not minimum <= value <= maximum:
        warnings.warn(
            f"{name} must be between {minimum} and {maximum}; using default {default}",
            RuntimeWarning,
        )
        return default
    return value


def yaw_mask_config() -> tuple[float, float, float, float]:
    """Return validated yaw adaptation and contour settings."""
    return (
        _env_float(YAW_THRESHOLD_ENV, 0.12, 0.0, 0.95),
        _env_float(MAX_SHRINK_ENV, 0.35, 0.0, 0.80),
        _env_float(MAX_SHIFT_ENV, 0.10, 0.0, 0.40),
        _env_float(CONTOUR_FEATHER_ENV, 0.02, 0.0, 0.10),
    )


def max_editable_coverage() -> float:
    """Return the maximum fraction of the aligned crop that may be edited."""
    return _env_float(MAX_EDITABLE_COVERAGE_ENV, 0.18, 0.01, 0.50)


def signed_polygon_area(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    return float(
        0.5 * (
            np.dot(points[:, 0], np.roll(points[:, 1], -1))
            - np.dot(points[:, 1], np.roll(points[:, 0], -1))
        )
    )


def polygon_self_intersects(points: np.ndarray) -> bool:
    """Return whether a closed polygon has non-adjacent edge crossings."""
    points = np.asarray(points, dtype=np.float64)

    def orientation(a, b, c):
        ab, ac = b - a, c - a
        return ab[0] * ac[1] - ab[1] * ac[0]

    def intersects(a, b, c, d):
        o1, o2 = orientation(a, b, c), orientation(a, b, d)
        o3, o4 = orientation(c, d, a), orientation(c, d, b)
        return ((o1 > 0 > o2) or (o2 > 0 > o1)) and (
            (o3 > 0 > o4) or (o4 > 0 > o3)
        )

    count = len(points)
    for i in range(count):
        for j in range(i + 1, count):
            if j in (i, i + 1) or (i == 0 and j == count - 1):
                continue
            if intersects(points[i], points[(i + 1) % count], points[j], points[(j + 1) % count]):
                return True
    return False


def validate_polygon(
    points: np.ndarray, height: int, width: int, minimum_points: int = 3
) -> tuple[bool, str, float]:
    """Validate contour coordinates and winding before rasterization."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < minimum_points:
        return False, "insufficient_points", 0.0
    if not np.isfinite(points).all():
        return False, "non_finite", 0.0
    if (
        np.any(points[:, 0] < 0)
        or np.any(points[:, 0] >= width)
        or np.any(points[:, 1] < 0)
        or np.any(points[:, 1] >= height)
    ):
        return False, "out_of_bounds", 0.0
    if polygon_self_intersects(points):
        return False, "self_intersection", signed_polygon_area(points)
    area = signed_polygon_area(points)
    if abs(area) < 1.0:
        return False, "zero_area", area
    if area >= 0.0:
        return False, "unexpected_winding", area
    return True, "ok", area


def estimate_yaw_from_landmarks(landmarks_2d_106: np.ndarray) -> float:
    """Estimate signed yaw from the detector's eye and nose landmarks."""
    landmarks = np.asarray(landmarks_2d_106, dtype=np.float32)
    if landmarks.shape != (106, 2) or not np.isfinite(landmarks).all():
        raise ValueError(
            f"Expected finite detector-space landmarks with shape (106, 2), got {landmarks.shape}"
        )
    left_eye = landmarks[[43, 48, 49, 51, 50]].mean(axis=0)
    right_eye = landmarks[101:106].mean(axis=0)
    nose = landmarks[[74, 77, 83, 86]].mean(axis=0)
    eye_vector = right_eye - left_eye
    eye_distance = float(np.linalg.norm(eye_vector))
    if eye_distance < 1e-6:
        raise ValueError("Cannot estimate yaw from coincident eye centers")
    eye_axis = eye_vector / eye_distance
    midpoint = (left_eye + right_eye) * 0.5
    return float(np.dot(nose - midpoint, eye_axis) / (eye_distance * 0.5))


def yaw_adaptation_amount(yaw: float, threshold: float) -> float:
    """Smoothly map yaw beyond the threshold to a saturated 0..1 amount."""
    if not np.isfinite(yaw):
        return 0.0
    threshold = float(np.clip(threshold, 0.0, 0.95))
    linear = np.clip(
        (abs(float(yaw)) - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0
    )
    return float(linear * linear * (3.0 - 2.0 * linear))


def facial_contour_mask(
    aligned_landmarks_106: np.ndarray,
    height: int,
    width: int,
    feather: float | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Rasterize a feathered visible-face polygon; one means editable is allowed."""
    import cv2

    if feather is None:
        feather = yaw_mask_config()[3]
    landmarks = np.asarray(aligned_landmarks_106, dtype=np.float32)
    if landmarks.shape != (106, 2) or not np.isfinite(landmarks).all():
        return torch.ones((1, height, width), device=device, dtype=dtype)
    points = landmarks[FACE_CONTOUR_INDICES]
    valid, _, _ = validate_polygon(
        points, height, width, minimum_points=len(FACE_CONTOUR_INDICES)
    )
    if not valid:
        return torch.ones((1, height, width), device=device, dtype=dtype)
    contour = np.zeros((height, width), dtype=np.float32)
    cv2.fillPoly(contour, [np.rint(points).astype(np.int32)], 1.0)
    feather_pixels = int(round(float(np.clip(feather, 0.0, 0.10)) * min(height, width)))
    if feather_pixels > 0:
        kernel = feather_pixels * 2 + 1
        contour = cv2.GaussianBlur(
            contour, (kernel, kernel), sigmaX=max(feather_pixels / 2.0, 0.1)
        )
    return torch.from_numpy(contour).to(device=device, dtype=dtype).unsqueeze(0)


def adapt_canonical_mask(
    canonical_mask: torch.Tensor,
    yaw: float,
    threshold: float | None = None,
    max_shrink: float | None = None,
    max_shift: float | None = None,
    contour_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply smooth yaw shrink/shift/attenuation and optional contour clipping."""
    configured = yaw_mask_config()
    threshold = configured[0] if threshold is None else threshold
    max_shrink = configured[1] if max_shrink is None else max_shrink
    max_shift = configured[2] if max_shift is None else max_shift
    threshold = float(np.clip(threshold, 0.0, 0.95))
    max_shrink = float(np.clip(max_shrink, 0.0, 0.80))
    max_shift = float(np.clip(max_shift, 0.0, 0.40))
    amount = yaw_adaptation_amount(yaw, threshold)

    original_shape = canonical_mask.shape
    mask = canonical_mask
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[None]
    if mask.ndim != 4:
        raise ValueError(f"Expected a 2D, CHW, or NCHW mask, got {original_shape}")
    editable = 1.0 - mask
    _, _, height, width = editable.shape

    if amount > 0.0:
        shrink = max_shrink * amount
        shift = np.sign(yaw) * max_shift * amount
        theta = editable.new_tensor(
            [[[1.0 / max(1.0 - shrink, 1e-4), 0.0, -2.0 * shift], [0.0, 1.0, 0.0]]]
        ).expand(editable.shape[0], -1, -1)
        grid = F.affine_grid(theta, editable.shape, align_corners=False)
        editable = F.grid_sample(
            editable, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        x = torch.linspace(-1.0, 1.0, width, device=editable.device, dtype=editable.dtype)
        silhouette_coordinate = x if yaw >= 0 else -x
        edge_ramp = ((1.0 - silhouette_coordinate) / 0.5).clamp(0.0, 1.0)
        attenuation = 1.0 - amount * (1.0 - edge_ramp)
        editable = editable * attenuation.view(1, 1, 1, width)

    if contour_mask is not None:
        contour = contour_mask.to(device=editable.device, dtype=editable.dtype)
        if contour.ndim == 2:
            contour = contour[None, None]
        elif contour.ndim == 3:
            contour = contour[None]
        if contour.shape[-2:] != editable.shape[-2:]:
            raise ValueError("Contour mask spatial dimensions must match the canonical mask")
        editable = editable * contour

    adapted = (1.0 - editable).clamp(0.0, 1.0)
    if len(original_shape) == 2:
        return adapted[0, 0]
    if len(original_shape) == 3:
        return adapted[0]
    return adapted
