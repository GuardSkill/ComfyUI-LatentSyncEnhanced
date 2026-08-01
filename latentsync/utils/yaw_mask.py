"""Continuous yaw-aware adaptation for the canonical LatentSync mouth mask."""

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
MOUTH_ROI_MODE_ENV = "LATENTSYNC_MOUTH_ROI_MODE"
MOUTH_ROI_HORIZONTAL_PADDING_ENV = "LATENTSYNC_MOUTH_ROI_HORIZONTAL_PADDING"
MOUTH_ROI_UPPER_PADDING_ENV = "LATENTSYNC_MOUTH_ROI_UPPER_PADDING"
MOUTH_ROI_LOWER_PADDING_ENV = "LATENTSYNC_MOUTH_ROI_LOWER_PADDING"
MOUTH_DILATION_FRACTION_ENV = "LATENTSYNC_MOUTH_DILATION_FRACTION"
MOUTH_FEATHER_FRACTION_ENV = "LATENTSYNC_MOUTH_FEATHER_FRACTION"
MAX_MOUTH_ROI_COVERAGE_ENV = "LATENTSYNC_MOUTH_ROI_MAX_COVERAGE"
MOUTH_ROI_HARD_MAX_COVERAGE = 0.14

# Compatibility aliases for launch scripts using the pre-mode spellings.
LEGACY_MOUTH_HORIZONTAL_PADDING_ENV = "LATENTSYNC_MOUTH_HORIZONTAL_PADDING"
LEGACY_MOUTH_UPPER_PADDING_ENV = "LATENTSYNC_MOUTH_UPPER_PADDING"
LEGACY_MOUTH_LOWER_PADDING_ENV = "LATENTSYNC_MOUTH_LOWER_PADDING"
LEGACY_MAX_MOUTH_ROI_COVERAGE_ENV = "LATENTSYNC_MOUTH_MAX_ROI_COVERAGE"
MOUTH_HORIZONTAL_PADDING_ENV = MOUTH_ROI_HORIZONTAL_PADDING_ENV
MOUTH_UPPER_PADDING_ENV = MOUTH_ROI_UPPER_PADDING_ENV
MOUTH_LOWER_PADDING_ENV = MOUTH_ROI_LOWER_PADDING_ENV

# InsightFace's 2d106 model uses the JD-landmark topology.  Its numeric IDs are
# not spatially sequential.  This is the complete silhouette walk, from one
# temple, around the jaw/chin, to the other temple.  The old value was only the
# sparse 17-point 106->68 conversion and was not a polygon topology.
FACE_CONTOUR_INDICES = [
    1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0,
    24, 23, 22, 21, 20, 19, 18, 32, 31, 30, 29, 28, 27, 26, 25, 17,
]
OLD_SPARSE_CONTOUR_INDICES = [1, 10, 12, 14, 16, 3, 5, 7, 0, 23, 21, 19, 32, 30, 28, 26, 17]
# InsightFace/JD 106 does not number the mouth points consecutively.  This is
# the outer-lip walk in image coordinates: left commissure, across the upper
# lip, right commissure, and back along the lower lip.  The old 20-point list
# appended the eight inner-lip points and was suitable only for a 106->68
# conversion; filling all of those points as a rectangle caused the ROI bug.
MOUTH_OUTER_LANDMARK_INDICES = [52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55]
MOUTH_INNER_LANDMARK_INDICES = [65, 66, 62, 70, 69, 57, 60, 54]
MOUTH_LANDMARK_INDICES = MOUTH_OUTER_LANDMARK_INDICES

# These are the two six-point eye contours in the same JD/InsightFace index
# mapping.  They are used only for the mouth ROI's vertical safety check.
EYE_LANDMARK_INDICES = [35, 41, 42, 39, 37, 36, 89, 95, 96, 93, 91, 90]


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



def mouth_roi_mode(value: str | None = None) -> str:
    """Return the selected ROI mode, defaulting to contextual expansion."""

    raw = os.getenv(MOUTH_ROI_MODE_ENV, "expanded") if value is None else value
    mode = str(raw).strip().lower()
    if mode not in {"tight", "expanded"}:
        warnings.warn(
            f"{MOUTH_ROI_MODE_ENV} must be tight or expanded; using expanded",
            RuntimeWarning,
        )
        return "expanded"
    return mode


def _env_float_with_legacy(
    name: str,
    legacy_name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """Read a new ROI-prefixed control before its legacy spelling."""

    if name in os.environ:
        return _env_float(name, default, minimum, maximum)
    if legacy_name in os.environ:
        return _env_float(legacy_name, default, minimum, maximum)
    return default


def yaw_mask_config() -> tuple[float, float, float, float]:
    """Return validated yaw-mask configuration values."""

    return (
        _env_float(YAW_THRESHOLD_ENV, 0.12, 0.0, 0.95),
        _env_float(MAX_SHRINK_ENV, 0.35, 0.0, 0.80),
        _env_float(MAX_SHIFT_ENV, 0.10, 0.0, 0.40),
        _env_float(CONTOUR_FEATHER_ENV, 0.02, 0.0, 0.10),
    )


def max_editable_coverage() -> float:
    """Maximum crop fraction that diffusion may edit (conservative mouth default)."""

    return _env_float(MAX_EDITABLE_COVERAGE_ENV, 0.18, 0.01, 0.50)


def mouth_roi_config(mode: str | None = None) -> dict[str, float | str]:
    """Return bounded mouth-local padding and rasterization settings.

    Padding values are fractions of the detected outer-lip width/height, not
    fractions of the aligned crop.  Dilation and feathering use the smaller
    of the mouth bounding width and height as their scale.
    """

    mode = mouth_roi_mode(mode)
    defaults = {
        "tight": {
            "horizontal_padding": 0.12,
            "upper_padding": 0.20,
            "lower_padding": 0.30,
            "max_coverage": 0.08,
        },
        "expanded": {
            "horizontal_padding": 0.24,
            "upper_padding": 0.28,
            "lower_padding": 0.45,
            "max_coverage": 0.12,
        },
    }[mode]
    return {
        "mode": mode,
        "horizontal_padding": _env_float_with_legacy(
            MOUTH_ROI_HORIZONTAL_PADDING_ENV,
            LEGACY_MOUTH_HORIZONTAL_PADDING_ENV,
            defaults["horizontal_padding"],
            0.0,
            0.40,
        ),
        "upper_padding": _env_float_with_legacy(
            MOUTH_ROI_UPPER_PADDING_ENV,
            LEGACY_MOUTH_UPPER_PADDING_ENV,
            defaults["upper_padding"],
            0.0,
            0.75,
        ),
        "lower_padding": _env_float_with_legacy(
            MOUTH_ROI_LOWER_PADDING_ENV,
            LEGACY_MOUTH_LOWER_PADDING_ENV,
            defaults["lower_padding"],
            0.0,
            0.75,
        ),
        "dilation_fraction": _env_float(
            MOUTH_DILATION_FRACTION_ENV, 0.06, 0.0, 0.12
        ),
        "feather_fraction": _env_float(
            MOUTH_FEATHER_FRACTION_ENV, 0.08, 0.0, 0.20
        ),
        "max_coverage": _env_float_with_legacy(
            MAX_MOUTH_ROI_COVERAGE_ENV,
            LEGACY_MAX_MOUTH_ROI_COVERAGE_ENV,
            defaults["max_coverage"],
            0.01,
            MOUTH_ROI_HARD_MAX_COVERAGE,
        ),
    }


def max_mouth_roi_coverage() -> float:
    """Maximum fractional area of a valid landmark-derived mouth ROI."""

    return float(mouth_roi_config()["max_coverage"])


def signed_polygon_area(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    return float(0.5 * (np.dot(points[:, 0], np.roll(points[:, 1], -1)) -
                        np.dot(points[:, 1], np.roll(points[:, 0], -1))))


def polygon_self_intersects(points: np.ndarray) -> bool:
    """Return whether a closed simple polygon has non-adjacent edge crossings."""

    points = np.asarray(points, dtype=np.float64)

    def orientation(a, b, c):
        ab, ac = b - a, c - a
        return ab[0] * ac[1] - ab[1] * ac[0]

    def intersects(a, b, c, d):
        o1, o2 = orientation(a, b, c), orientation(a, b, d)
        o3, o4 = orientation(c, d, a), orientation(c, d, b)
        return ((o1 > 0 > o2) or (o2 > 0 > o1)) and ((o3 > 0 > o4) or (o4 > 0 > o3))

    count = len(points)
    for i in range(count):
        for j in range(i + 1, count):
            if j in (i, i + 1) or (i == 0 and j == count - 1):
                continue
            if intersects(points[i], points[(i + 1) % count], points[j], points[(j + 1) % count]):
                return True
    return False


def validate_polygon(points: np.ndarray, height: int, width: int, minimum_points: int = 3) -> tuple[bool, str, float]:
    """Validate coordinates and topology before passing them to OpenCV."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < minimum_points:
        return False, "insufficient_points", 0.0
    if not np.isfinite(points).all():
        return False, "non_finite", 0.0
    if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height):
        return False, "out_of_bounds", 0.0
    if polygon_self_intersects(points):
        return False, "self_intersection", signed_polygon_area(points)
    area = signed_polygon_area(points)
    if abs(area) < 1.0:
        return False, "zero_area", area
    # This topology is defined clockwise in image coordinates (negative area
    # in Cartesian coordinates because image Y increases downward). A reversed or
    # scrambled detector result is rejected rather than silently re-ordered.
    if area >= 0.0:
        return False, "unexpected_winding", area
    return True, "ok", area


def validate_mouth_polygon(
    points: np.ndarray,
    height: int,
    width: int,
    minimum_points: int = len(MOUTH_OUTER_LANDMARK_INDICES),
) -> tuple[bool, str, float]:
    """Validate the fixed outer-lip walk before rasterization.

    The JD mouth walk is clockwise in image coordinates, so a positive
    shoelace area is expected here.  Unlike the face-contour validator, this
    validator intentionally does not reorder points: a scrambled detector
    result must fall back safely instead of creating a plausible-looking but
    incorrect polygon.
    """

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < minimum_points:
        return False, "insufficient_points", 0.0
    if not np.isfinite(points).all():
        return False, "non_finite", 0.0
    if (
        len(np.unique(points, axis=0)) < minimum_points
        or
        np.any(points[:, 0] < 0)
        or np.any(points[:, 0] >= width)
        or np.any(points[:, 1] < 0)
        or np.any(points[:, 1] >= height)
    ):
        return False, "out_of_bounds", signed_polygon_area(points)
    if polygon_self_intersects(points):
        return False, "self_intersection", signed_polygon_area(points)
    area = signed_polygon_area(points)
    if abs(area) < 1.0:
        return False, "zero_area", area
    if area <= 0.0:
        return False, "unexpected_winding", area
    return True, "ok", area


def closed_mouth_polygon(aligned_landmarks_106: np.ndarray) -> np.ndarray:
    """Return the selected outer-lip points with the first point repeated."""

    landmarks = np.asarray(aligned_landmarks_106, dtype=np.float32)
    if landmarks.shape != (106, 2):
        raise ValueError(f"Expected aligned landmarks with shape (106, 2), got {landmarks.shape}")
    points = landmarks[MOUTH_OUTER_LANDMARK_INDICES].copy()
    return np.concatenate([points, points[:1]], axis=0)


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    weights = np.asarray(mask, dtype=np.float64)
    total = float(weights.sum())
    if total <= 1e-8:
        return None
    y, x = np.indices(weights.shape, dtype=np.float64)
    return float((weights * x).sum() / total), float((weights * y).sum() / total)


def _mask_bounds(mask: np.ndarray, threshold: float = 1e-6) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(np.asarray(mask) > threshold)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _empty_mouth_roi_geometry(height: int, width: int, reason: str) -> dict:
    zeros = np.zeros((height, width), dtype=np.float32)
    return {
        "valid": False,
        "reason": reason,
        "mask": zeros,
        "raw_polygon": zeros.copy(),
        "dilated_roi": zeros.copy(),
        "feathered_roi": zeros.copy(),
        "outer_points": np.empty((0, 2), dtype=np.float32),
        "closed_points": np.empty((0, 2), dtype=np.float32),
        "mouth_centroid": None,
        "roi_centroid": None,
        "mouth_width": 0.0,
        "mouth_height": 0.0,
        "dilation_radius": 0,
        "feather_radius": 0,
        "raw_coverage": 0.0,
        "dilated_coverage": 0.0,
        "feathered_coverage": 0.0,
        "roi_bounds": None,
    }


def mouth_roi_geometry(
    aligned_landmarks_106,
    height: int,
    width: int,
    *,
    mode: str | None = None,
    horizontal_padding: float | None = None,
    upper_padding: float | None = None,
    lower_padding: float | None = None,
    dilation_fraction: float | None = None,
    feather_fraction: float | None = None,
    max_coverage: float | None = None,
) -> dict:
    """Build a bounded, feathered, landmark-shaped mouth ROI.

    The returned masks use geometry polarity (``1`` means editable/allowed):
    ``raw_polygon`` is the closed outer-lip polygon, ``dilated_roi`` is its
    bounded padded/dilated form, and ``feathered_roi`` is the final soft ROI.
    Invalid geometry returns a zero ``mask`` and a reason; callers that have a
    canonical production mask can then choose that explicit safe fallback.
    """

    import cv2

    if int(height) <= 0 or int(width) <= 0:
        return _empty_mouth_roi_geometry(max(int(height), 1), max(int(width), 1), "invalid_crop")

    config = mouth_roi_config(mode=mode)
    horizontal_padding = config["horizontal_padding"] if horizontal_padding is None else float(horizontal_padding)
    upper_padding = config["upper_padding"] if upper_padding is None else float(upper_padding)
    lower_padding = config["lower_padding"] if lower_padding is None else float(lower_padding)
    dilation_fraction = config["dilation_fraction"] if dilation_fraction is None else float(dilation_fraction)
    feather_fraction = config["feather_fraction"] if feather_fraction is None else float(feather_fraction)
    max_coverage = config["max_coverage"] if max_coverage is None else float(max_coverage)
    max_coverage = min(max_coverage, MOUTH_ROI_HARD_MAX_COVERAGE)

    landmarks = np.asarray(aligned_landmarks_106, dtype=np.float32)
    if landmarks.shape != (106, 2) or not np.isfinite(landmarks).all():
        return _empty_mouth_roi_geometry(height, width, "invalid_landmarks")

    outer_points = landmarks[MOUTH_OUTER_LANDMARK_INDICES].copy()
    valid, reason, _ = validate_mouth_polygon(outer_points, height, width)
    if not valid:
        geometry = _empty_mouth_roi_geometry(height, width, reason)
        geometry["outer_points"] = outer_points
        geometry["closed_points"] = np.concatenate([outer_points, outer_points[:1]], axis=0)
        return geometry

    x0, y0 = outer_points.min(axis=0)
    x1, y1 = outer_points.max(axis=0)
    mouth_width = float(x1 - x0)
    mouth_height = float(y1 - y0)
    mouth_centroid = tuple(float(value) for value in outer_points.mean(axis=0))

    raw_polygon = np.zeros((height, width), dtype=np.float32)
    closed_points = np.concatenate([outer_points, outer_points[:1]], axis=0)
    cv2.fillPoly(raw_polygon, [np.rint(closed_points).astype(np.int32)], 1.0)

    # Padding is deliberately applied in mouth-relative coordinates.  The
    # horizontal setting is per side; upper/lower settings extend the two
    # vertical bounds independently while preserving the lip topology.
    pad_x = float(np.clip(horizontal_padding, 0.0, 0.40))
    pad_upper = float(np.clip(upper_padding, 0.0, 0.75))
    pad_lower = float(np.clip(lower_padding, 0.0, 0.75))
    x_center = (float(x0) + float(x1)) * 0.5
    padded_points = outer_points.copy()
    padded_points[:, 0] = x_center + (padded_points[:, 0] - x_center) * (1.0 + 2.0 * pad_x)
    padded_points[:, 1] = float(y0) - pad_upper * mouth_height + (
        (padded_points[:, 1] - float(y0)) / mouth_height
    ) * (mouth_height * (1.0 + pad_upper + pad_lower))
    padded_points[:, 0] = np.clip(padded_points[:, 0], 0.0, width - 1.0)
    padded_points[:, 1] = np.clip(padded_points[:, 1], 0.0, height - 1.0)

    scale = max(min(mouth_width, mouth_height), 1.0)
    max_dilation_radius = max(1, int(round(0.12 * scale)))
    dilation_radius = int(np.clip(round(float(dilation_fraction) * scale), 1, max_dilation_radius))
    padded_polygon = np.zeros((height, width), dtype=np.float32)
    padded_closed = np.concatenate([padded_points, padded_points[:1]], axis=0)
    cv2.fillPoly(padded_polygon, [np.rint(padded_closed).astype(np.int32)], 1.0)
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation_radius + 1, 2 * dilation_radius + 1)
    )
    dilated_roi = cv2.dilate(padded_polygon, dilation_kernel, iterations=1)

    max_feather_radius = max(1, int(round(0.20 * scale)))
    feather_radius = int(np.clip(round(float(feather_fraction) * scale), 1, max_feather_radius))
    feather_kernel = 2 * feather_radius + 1
    feathered_roi = cv2.GaussianBlur(
        dilated_roi,
        (feather_kernel, feather_kernel),
        sigmaX=max(feather_radius / 2.0, 0.5),
    ).astype(np.float32)
    feathered_roi = np.clip(feathered_roi, 0.0, 1.0)

    raw_coverage = float(raw_polygon.mean())
    dilated_coverage = float(dilated_roi.mean())
    feathered_coverage = float(feathered_roi.mean())
    roi_centroid = _mask_centroid(feathered_roi)
    roi_bounds = _mask_bounds(feathered_roi)
    geometry = {
        "valid": True,
        "reason": "ok",
        "mask": feathered_roi.copy(),
        "raw_polygon": raw_polygon,
        "dilated_roi": dilated_roi.astype(np.float32),
        "feathered_roi": feathered_roi,
        "outer_points": outer_points,
        "closed_points": closed_points,
        "padded_points": padded_points,
        "mouth_centroid": mouth_centroid,
        "roi_centroid": roi_centroid,
        "mouth_width": mouth_width,
        "mouth_height": mouth_height,
        "dilation_radius": dilation_radius,
        "feather_radius": feather_radius,
        "raw_coverage": raw_coverage,
        "dilated_coverage": dilated_coverage,
        "feathered_coverage": feathered_coverage,
        "roi_bounds": roi_bounds,
    }

    # The support checks below make the ROI fail closed.  A detector point
    # outside the mouth cannot enlarge this mask because no non-mouth point is
    # ever used in its construction.
    if feathered_coverage > float(np.clip(max_coverage, 0.01, 0.25)):
        geometry["valid"] = False
        geometry["reason"] = "coverage_limit"
    elif roi_centroid is None:
        geometry["valid"] = False
        geometry["reason"] = "empty_roi"
    else:
        dx = abs(roi_centroid[0] - mouth_centroid[0])
        dy = abs(roi_centroid[1] - mouth_centroid[1])
        if dx > max(0.35 * mouth_width, 2.0) or dy > max(0.50 * mouth_height, 2.0):
            geometry["valid"] = False
            geometry["reason"] = "centroid_drift"

    if geometry["valid"] and roi_bounds is not None:
        _, roi_top, _, roi_bottom = roi_bounds
        eye_points = landmarks[EYE_LANDMARK_INDICES]
        if np.ptp(eye_points, axis=0).max() > 1.0 and np.max(np.abs(eye_points)) > 1.0:
            eye_bottom = float(np.max(eye_points[:, 1]))
            if roi_top < eye_bottom:
                geometry["valid"] = False
                geometry["reason"] = "reaches_eyes"

        contour_points = landmarks[FACE_CONTOUR_INDICES]
        if geometry["valid"] and np.ptp(contour_points, axis=0).max() > 1.0:
            chin_y = float(np.max(contour_points[:, 1]))
            if roi_bottom > chin_y + max(2.0, 0.10 * mouth_height):
                geometry["valid"] = False
                geometry["reason"] = "below_chin"

    if not geometry["valid"]:
        geometry["mask"] = np.zeros((height, width), dtype=np.float32)
    return geometry


def mouth_roi_mask(
    aligned_landmarks_106,
    height,
    width,
    device="cpu",
    dtype=torch.float32,
    **kwargs,
) -> torch.Tensor:
    """Return the feathered landmark-shaped ROI; 1 means editable/allowed."""

    geometry = mouth_roi_geometry(aligned_landmarks_106, height, width, **kwargs)
    return torch.from_numpy(geometry["mask"]).to(device=device, dtype=dtype).unsqueeze(0)


def estimate_yaw_from_landmarks(landmarks_2d_106: np.ndarray) -> float:
    """Estimate signed yaw from the existing eye and nose landmark groups.

    The score is normalized by half the inter-eye distance. Positive values mean
    that the nose has moved toward image-right; negative values mean image-left.
    """

    landmarks = np.asarray(landmarks_2d_106, dtype=np.float32)
    if landmarks.shape != (106, 2) or not np.isfinite(landmarks).all():
        raise ValueError(f"Expected finite detector-space landmarks with shape (106, 2), got {landmarks.shape}")
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
    """Smoothly map absolute yaw from the threshold to a saturated 0..1 amount."""

    if not np.isfinite(yaw):
        warnings.warn("Non-finite yaw estimate; treating frame as frontal", RuntimeWarning)
        return 0.0
    threshold = float(np.clip(threshold, 0.0, 0.95))
    linear = np.clip((abs(float(yaw)) - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    return float(linear * linear * (3.0 - 2.0 * linear))



def transform_landmarks_to_aligned(
    landmarks_2d_106: np.ndarray,
    affine_matrix: np.ndarray | torch.Tensor,
    aligned_size: tuple[int, int],
    resolution: int,
) -> np.ndarray:
    """Map original-frame landmarks through the existing affine crop transform."""

    landmarks = np.asarray(landmarks_2d_106, dtype=np.float32)
    if isinstance(affine_matrix, torch.Tensor):
        matrix = affine_matrix.detach().cpu().numpy()
    else:
        matrix = np.asarray(affine_matrix)
    matrix = matrix.reshape(-1, 2, 3)[0].astype(np.float32)
    homogeneous = np.concatenate([landmarks, np.ones((len(landmarks), 1), np.float32)], axis=1)
    aligned = homogeneous @ matrix.T
    width, height = aligned_size
    aligned[:, 0] *= resolution / float(width)
    aligned[:, 1] *= resolution / float(height)
    return aligned


def facial_contour_mask(
    aligned_landmarks_106: np.ndarray,
    height: int,
    width: int,
    feather: float | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a softly feathered visible-face polygon mask in aligned space."""

    import cv2

    if feather is None:
        feather = yaw_mask_config()[3]
    feather = float(np.clip(feather, 0.0, 0.10))
    landmarks = np.asarray(aligned_landmarks_106, dtype=np.float32)
    if landmarks.shape != (106, 2) or not np.isfinite(landmarks).all():
        warnings.warn("Invalid aligned landmarks; contour clipping disabled for frame", RuntimeWarning)
        return torch.ones((1, height, width), device=device, dtype=dtype)
    points = landmarks[FACE_CONTOUR_INDICES].copy()
    contour = np.zeros((height, width), dtype=np.float32)
    valid, reason, _ = validate_polygon(points, height, width, minimum_points=len(FACE_CONTOUR_INDICES))
    if not valid:
        warnings.warn(f"Invalid facial contour ({reason}); contour clipping disabled for frame", RuntimeWarning)
        return torch.ones((1, height, width), device=device, dtype=dtype)
    cv2.fillPoly(contour, [np.rint(points).astype(np.int32)], 1.0)
    feather_pixels = int(round(float(feather) * min(height, width)))
    if feather_pixels > 0:
        kernel = feather_pixels * 2 + 1
        contour = cv2.GaussianBlur(contour, (kernel, kernel), sigmaX=max(feather_pixels / 2.0, 0.1))
    return torch.from_numpy(contour).to(device=device, dtype=dtype).unsqueeze(0)


def adapt_canonical_mask(
    canonical_mask: torch.Tensor,
    yaw: float,
    threshold: float | None = None,
    max_shrink: float | None = None,
    max_shift: float | None = None,
    contour_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shrink, shift, and softly attenuate the editable (black) mouth region."""

    configured = yaw_mask_config()
    threshold = configured[0] if threshold is None else threshold
    max_shrink = configured[1] if max_shrink is None else max_shrink
    max_shift = configured[2] if max_shift is None else max_shift
    threshold = float(np.clip(threshold, 0.0, 0.95))
    max_shrink = float(np.clip(max_shrink, 0.0, 0.80))
    max_shift = float(np.clip(max_shift, 0.0, 0.40))
    amount = yaw_adaptation_amount(yaw, threshold)
    if amount == 0.0:
        adapted = canonical_mask.clone()
        if contour_mask is None:
            return adapted
        editable = (1.0 - adapted) * contour_mask.to(device=adapted.device, dtype=adapted.dtype)
        return (1.0 - editable).clamp(0.0, 1.0)

    original_shape = canonical_mask.shape
    mask = canonical_mask
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[None]
    if mask.ndim != 4:
        raise ValueError(f"Expected a 2D, CHW, or NCHW mask, got {original_shape}")

    # Work in editable-mask polarity: canonical white is preserved context and
    # canonical black is the diffusion/composition mouth region.
    editable = 1.0 - mask
    _, _, height, width = editable.shape
    shrink = max_shrink * amount
    shift = np.sign(yaw) * max_shift * amount
    theta = editable.new_tensor([[[1.0 / max(1.0 - shrink, 1e-4), 0.0, -2.0 * shift], [0.0, 1.0, 0.0]]])
    theta = theta.expand(editable.shape[0], -1, -1)
    grid = F.affine_grid(theta, editable.shape, align_corners=False)
    editable = F.grid_sample(editable, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # Suppress the outer image edge on the side toward which a profile points.
    # This is deliberately soft and continuous; it is not a contour clip.
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
