import unittest
from unittest.mock import patch

import numpy as np
import torch

from latentsync.utils.yaw_mask import (
    EYE_LANDMARK_INDICES,
    FACE_CONTOUR_INDICES,
    MOUTH_LANDMARK_INDICES,
    MOUTH_OUTER_LANDMARK_INDICES,
    adapt_canonical_mask,
    closed_mouth_polygon,
    estimate_yaw_from_landmarks,
    facial_contour_mask,
    max_editable_coverage,
    max_mouth_roi_coverage,
    mouth_roi_geometry,
    mouth_roi_mask,
    polygon_self_intersects,
    signed_polygon_area,
    validate_mouth_polygon,
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
    landmarks[MOUTH_OUTER_LANDMARK_INDICES] = np.array(
        [
            [24, 40], [28, 38], [32, 38], [36, 38], [40, 40], [40, 44],
            [36, 47], [32, 48], [28, 47], [24, 44], [27, 41], [32, 40],
        ],
        dtype=np.float32,
    )
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
                1.0
                - (
                    (1.0 - adapt_canonical_mask(processor.mask_image, yaw))
                    * mouth_roi_mask(landmark, 64, 64)
                    * facial_contour_mask(landmark, 64, 64)
                )
                for yaw, landmark in zip(yaws, landmarks)
            ]
        )
        self.assertTrue(torch.allclose(masks, expected))
        self.assertFalse(torch.allclose(masks[0], masks[1]))
        self.assertFalse(torch.allclose(masks[1], masks[2]))

    def test_outer_lip_ordering_closes_a_landmark_shaped_polygon(self):
        self.assertEqual(
            MOUTH_LANDMARK_INDICES,
            [52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55],
        )
        self.assertEqual(MOUTH_LANDMARK_INDICES, MOUTH_OUTER_LANDMARK_INDICES)
        landmarks = synthetic_contour_landmarks()
        closed = closed_mouth_polygon(landmarks)
        self.assertEqual(tuple(closed.shape), (13, 2))
        self.assertTrue(np.array_equal(closed[0], closed[-1]))
        self.assertGreater(signed_polygon_area(closed[:-1]), 0.0)
        self.assertFalse(polygon_self_intersects(closed[:-1]))
        self.assertEqual(
            validate_mouth_polygon(landmarks[MOUTH_LANDMARK_INDICES], 64, 64)[0:2],
            (True, "ok"),
        )

        geometry = mouth_roi_geometry(landmarks, 64, 64)
        x0, y0, x1, y1 = geometry["roi_bounds"]
        bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
        self.assertLess(float(geometry["raw_polygon"].sum()), 0.85 * bbox_area)

    def test_mouth_roi_uses_relative_padding_bounded_dilation_and_feathering(self):
        landmarks = synthetic_contour_landmarks()
        tight = mouth_roi_geometry(
            landmarks,
            64,
            64,
            horizontal_padding=0.0,
            upper_padding=0.0,
            lower_padding=0.0,
            dilation_fraction=0.0,
            feather_fraction=0.0,
        )
        padded = mouth_roi_geometry(
            landmarks,
            64,
            64,
            horizontal_padding=0.12,
            upper_padding=0.20,
            lower_padding=0.30,
            dilation_fraction=0.12,
            feather_fraction=0.20,
        )
        self.assertTrue(tight["valid"], tight["reason"])
        self.assertTrue(padded["valid"], padded["reason"])
        self.assertGreater(padded["dilated_coverage"], tight["dilated_coverage"])
        self.assertGreater(padded["dilation_radius"], 0)
        self.assertGreater(padded["feather_radius"], 0)
        self.assertTrue(
            np.any((padded["feathered_roi"] > 0.0) & (padded["feathered_roi"] < 1.0)
        ))
        self.assertLessEqual(
            padded["feathered_coverage"], max_mouth_roi_coverage()
        )

    def test_mouth_roi_centroid_eye_chin_and_coverage_safety_checks(self):
        landmarks = synthetic_contour_landmarks()
        geometry = mouth_roi_geometry(landmarks, 64, 64)
        self.assertTrue(geometry["valid"], geometry["reason"])
        mouth_x, mouth_y = geometry["mouth_centroid"]
        roi_x, roi_y = geometry["roi_centroid"]
        self.assertLessEqual(
            abs(roi_x - mouth_x), max(0.35 * geometry["mouth_width"], 2.0)
        )
        self.assertLessEqual(
            abs(roi_y - mouth_y), max(0.50 * geometry["mouth_height"], 2.0)
        )

        eye_collision = landmarks.copy()
        eye_collision[EYE_LANDMARK_INDICES] = np.array(
            [
                [20, 42], [24, 42], [28, 42], [32, 42], [36, 42], [40, 42],
                [24, 42], [28, 42], [32, 42], [36, 42], [40, 42], [44, 42],
            ],
            dtype=np.float32,
        )
        eye_geometry = mouth_roi_geometry(eye_collision, 64, 64)
        self.assertFalse(eye_geometry["valid"])
        self.assertEqual(eye_geometry["reason"], "reaches_eyes")
        self.assertEqual(float(eye_geometry["mask"].sum()), 0.0)

        chin_collision = landmarks.copy()
        chin_collision[FACE_CONTOUR_INDICES] = np.linspace(
            [10, 20], [54, 40], len(FACE_CONTOUR_INDICES)
        )
        chin_geometry = mouth_roi_geometry(chin_collision, 64, 64)
        self.assertFalse(chin_geometry["valid"])
        self.assertEqual(chin_geometry["reason"], "below_chin")

        capped = mouth_roi_geometry(landmarks, 64, 64, max_coverage=0.01)
        self.assertFalse(capped["valid"])
        self.assertEqual(capped["reason"], "coverage_limit")
        self.assertEqual(float(capped["mask"].sum()), 0.0)

    def test_invalid_mouth_geometry_is_zero_and_never_a_rectangle(self):
        landmarks = synthetic_contour_landmarks()
        landmarks[MOUTH_LANDMARK_INDICES[3]] = landmarks[MOUTH_LANDMARK_INDICES[0]]
        geometry = mouth_roi_geometry(landmarks, 64, 64)
        self.assertFalse(geometry["valid"])
        self.assertEqual(float(geometry["mask"].sum()), 0.0)
        self.assertEqual(float(mouth_roi_mask(landmarks, 64, 64).sum()), 0.0)

    def test_nearby_non_mouth_landmarks_do_not_expand_the_roi(self):
        landmarks = synthetic_contour_landmarks()
        baseline = mouth_roi_mask(landmarks, 64, 64)
        protected = set(MOUTH_LANDMARK_INDICES) | set(FACE_CONTOUR_INDICES)
        unrelated = [index for index in range(106) if index not in protected]
        landmarks[unrelated] = np.array([31.0, 43.0], dtype=np.float32)
        self.assertTrue(torch.equal(baseline, mouth_roi_mask(landmarks, 64, 64)))

    def test_image_processor_keeps_final_editable_strength_in_all_source_subsets(self):
        from latentsync.utils.image_processor import ImageProcessor

        processor = ImageProcessor.__new__(ImageProcessor)
        processor.mask_image = canonical_test_mask()
        processor.resize = lambda image: image
        processor.normalize = lambda image: image
        landmarks = synthetic_contour_landmarks()
        _, _, mask = processor.preprocess_fixed_mask_image(
            torch.zeros(3, 64, 64), yaw=0.7, aligned_landmarks=landmarks
        )
        final_strength = 1.0 - mask
        yaw_strength = 1.0 - adapt_canonical_mask(processor.mask_image, 0.7)
        mouth_strength = mouth_roi_mask(landmarks, 64, 64)
        contour_strength = facial_contour_mask(landmarks, 64, 64)
        self.assertTrue(torch.all(final_strength <= yaw_strength + 1e-6))
        self.assertTrue(torch.all(final_strength <= mouth_strength + 1e-6))
        self.assertTrue(torch.all(final_strength <= contour_strength + 1e-6))

    def test_image_processor_uses_canonical_fallback_for_invalid_mouth_geometry(self):
        from latentsync.utils.image_processor import ImageProcessor

        processor = ImageProcessor.__new__(ImageProcessor)
        processor.mask_image = canonical_test_mask()
        processor.mask_image[:, 38:40, 20:24] = 0.25
        processor.resize = lambda image: image
        processor.normalize = lambda image: image
        landmarks = synthetic_contour_landmarks()
        landmarks[MOUTH_LANDMARK_INDICES[3]] = landmarks[MOUTH_LANDMARK_INDICES[0]]
        _, _, mask = processor.preprocess_fixed_mask_image(
            torch.zeros(3, 64, 64), yaw=0.0, aligned_landmarks=landmarks
        )
        expected_editable = (
            (1.0 - processor.mask_image)
            * facial_contour_mask(landmarks, 64, 64)
        )
        self.assertTrue(torch.allclose(1.0 - mask, expected_editable, atol=1e-6))

    def test_editable_coverage_default_is_bounded(self):
        self.assertEqual(max_editable_coverage(), 0.18)


if __name__ == "__main__":
    unittest.main()
