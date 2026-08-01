import unittest
from unittest.mock import patch

import numpy as np
import torch

from latentsync.utils.yaw_mask import (
    FACE_CONTOUR_INDICES,
    adapt_canonical_mask,
    estimate_yaw_from_landmarks,
    facial_contour_mask,
    max_editable_coverage,
    polygon_self_intersects,
    signed_polygon_area,
    validate_polygon,
    yaw_adaptation_amount,
    yaw_mask_config,
)


def canonical_test_mask(size=64):
    mask = torch.ones(1, size, size)
    mask[:, 36:54, 14:50] = 0.0
    return mask


def editable_centroid(mask):
    editable = 1.0 - mask
    x = torch.arange(mask.shape[-1], dtype=mask.dtype)
    return float((editable.sum(dim=-2) * x).sum() / editable.sum())


def synthetic_landmarks(nose_x):
    landmarks = np.zeros((106, 2), dtype=np.float32)
    landmarks[[43, 48, 49, 51, 50]] = (20.0, 20.0)
    landmarks[101:106] = (44.0, 20.0)
    landmarks[[74, 77, 83, 86]] = (nose_x, 32.0)
    return landmarks


def synthetic_contour_landmarks(size=64, left=10, right=54, top=8, bottom=60):
    landmarks = np.zeros((106, 2), dtype=np.float32)
    count = len(FACE_CONTOUR_INDICES)
    left_side = np.linspace([left, top], [left, bottom], count // 2 + 1)
    right_side = np.linspace([right, bottom], [right, top], count - len(left_side) + 2)[1:]
    landmarks[FACE_CONTOUR_INDICES] = np.concatenate([left_side, right_side], axis=0)[:count]
    return landmarks


class YawContourProductionTests(unittest.TestCase):
    def test_environment_configuration_is_read(self):
        environment = {
            "LATENTSYNC_MOUTH_YAW_THRESHOLD": "0.2",
            "LATENTSYNC_MOUTH_MAX_HORIZONTAL_SHRINK": "0.3",
            "LATENTSYNC_MOUTH_MAX_HORIZONTAL_SHIFT": "0.15",
            "LATENTSYNC_MOUTH_CONTOUR_FEATHER": "0.03",
        }
        with patch.dict("os.environ", environment, clear=True):
            self.assertEqual(yaw_mask_config(), (0.2, 0.3, 0.15, 0.03))

    def test_invalid_configuration_falls_back(self):
        environment = {
            "LATENTSYNC_MOUTH_YAW_THRESHOLD": "invalid",
            "LATENTSYNC_MOUTH_MAX_HORIZONTAL_SHRINK": "2",
            "LATENTSYNC_MOUTH_MAX_HORIZONTAL_SHIFT": "nan",
            "LATENTSYNC_MOUTH_CONTOUR_FEATHER": "-1",
        }
        with patch.dict("os.environ", environment, clear=True):
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(yaw_mask_config(), (0.12, 0.35, 0.10, 0.02))

    def test_landmark_yaw_is_signed(self):
        frontal = estimate_yaw_from_landmarks(synthetic_landmarks(32.0))
        left = estimate_yaw_from_landmarks(synthetic_landmarks(24.0))
        right = estimate_yaw_from_landmarks(synthetic_landmarks(40.0))
        self.assertAlmostEqual(frontal, 0.0)
        self.assertLess(left, 0.0)
        self.assertGreater(right, 0.0)
        self.assertAlmostEqual(abs(left), abs(right))

    def test_smooth_adaptation_is_continuous_and_monotonic(self):
        yaws = [np.sin(np.deg2rad(angle)) for angle in (30, 45, 60)]
        strengths = [yaw_adaptation_amount(yaw, threshold=0.1) for yaw in yaws]
        self.assertGreater(strengths[0], 0.0)
        self.assertLess(strengths[0], strengths[1])
        self.assertLess(strengths[1], strengths[2])
        self.assertLess(strengths[2], 1.0)

    def test_profile_shrink_and_shift_follow_yaw_sign(self):
        mask = canonical_test_mask()
        left = adapt_canonical_mask(mask, yaw=-1.0, threshold=0.1, max_shrink=0.4, max_shift=0.15)
        right = adapt_canonical_mask(mask, yaw=1.0, threshold=0.1, max_shrink=0.4, max_shift=0.15)
        self.assertLess(float((1.0 - left).sum()), float((1.0 - mask).sum()))
        self.assertLess(float((1.0 - right).sum()), float((1.0 - mask).sum()))
        self.assertLess(editable_centroid(left), editable_centroid(mask))
        self.assertGreater(editable_centroid(right), editable_centroid(mask))

    def test_adapted_mask_preserves_shape_dtype_and_range(self):
        mask = canonical_test_mask().to(dtype=torch.float32)
        adapted = adapt_canonical_mask(mask, yaw=0.8)
        self.assertEqual(adapted.shape, mask.shape)
        self.assertEqual(adapted.dtype, mask.dtype)
        self.assertTrue(torch.isfinite(adapted).all())
        self.assertGreaterEqual(float(adapted.min()), 0.0)
        self.assertLessEqual(float(adapted.max()), 1.0)

    def test_contour_polygon_is_valid_and_clips_outside_editable_pixels(self):
        landmarks = synthetic_contour_landmarks(left=16, right=48)
        contour = facial_contour_mask(landmarks, 64, 64, feather=0.03)
        final_mask = adapt_canonical_mask(torch.zeros(1, 64, 64), yaw=0.0, contour_mask=contour)
        editable = 1.0 - final_mask
        self.assertEqual(float(editable[:, :, 0].max()), 0.0)
        self.assertGreater(float(editable[:, 38:49, 24:41].mean()), 0.8)
        self.assertTrue(torch.any((editable > 0.0) & (editable < 1.0)))

    def test_contour_topology_and_self_intersection_validation(self):
        landmarks = synthetic_contour_landmarks()
        points = landmarks[FACE_CONTOUR_INDICES]
        self.assertLess(signed_polygon_area(points), 0.0)
        self.assertFalse(polygon_self_intersects(points))
        self.assertEqual(validate_polygon(points, 64, 64)[0:2], (True, "ok"))
        bow_tie = np.array([[10, 10], [50, 50], [10, 50], [50, 10]], np.float32)
        self.assertTrue(polygon_self_intersects(bow_tie))
        self.assertEqual(validate_polygon(bow_tie, 64, 64)[1], "self_intersection")

    def test_contour_intersection_never_increases_editable_coverage(self):
        canonical = canonical_test_mask()
        contour = facial_contour_mask(synthetic_contour_landmarks(), 64, 64)
        for yaw in (0.0, -0.85, 0.85):
            yaw_only = adapt_canonical_mask(canonical, yaw)
            clipped = adapt_canonical_mask(canonical, yaw, contour_mask=contour)
            self.assertTrue(torch.all((1.0 - clipped) <= (1.0 - yaw_only) + 1e-6))
            self.assertLessEqual(float((1.0 - clipped).mean()), float((1.0 - yaw_only).mean()))

    def test_image_processor_applies_per_frame_yaw_and_contour(self):
        from latentsync.utils.image_processor import ImageProcessor

        processor = ImageProcessor.__new__(ImageProcessor)
        processor.mask_image = canonical_test_mask()
        processor.resize = lambda image: image
        processor.normalize = lambda image: image
        images = torch.zeros(3, 3, 64, 64)
        yaws = [-0.8, 0.0, 0.8]
        landmarks = [synthetic_contour_landmarks()] * 3
        _, _, masks = processor.prepare_masks_and_masked_images(
            images, yaws=yaws, aligned_landmarks=landmarks
        )
        expected = torch.stack(
            [
                adapt_canonical_mask(
                    processor.mask_image,
                    yaw,
                    contour_mask=facial_contour_mask(landmark, 64, 64),
                )[0:1]
                for yaw, landmark in zip(yaws, landmarks)
            ]
        )
        self.assertTrue(torch.allclose(masks, expected))
        self.assertFalse(torch.allclose(masks[0], masks[1]))
        self.assertFalse(torch.allclose(masks[1], masks[2]))

    def test_editable_coverage_default_is_bounded(self):
        self.assertEqual(max_editable_coverage(), 0.18)


if __name__ == "__main__":
    unittest.main()
