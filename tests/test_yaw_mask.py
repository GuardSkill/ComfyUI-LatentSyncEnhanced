import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import numpy as np

from latentsync.utils.yaw_mask import (
    FACE_CONTOUR_INDICES,
    MOUTH_INNER_LANDMARK_INDICES,
    MOUTH_LANDMARK_INDICES,
    MOUTH_OUTER_LANDMARK_INDICES,
    adapt_canonical_mask,
    closed_mouth_polygon,
    estimate_yaw_from_landmarks,
    facial_contour_mask,
    max_editable_coverage,
    max_mouth_roi_coverage,
    mouth_roi_config,
    mouth_roi_geometry,
    mouth_roi_mask,
    mouth_roi_mode,
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
    polygon = np.concatenate([left_side, right_side], axis=0)[:count]
    landmarks[FACE_CONTOUR_INDICES] = polygon
    mouth_outer = np.array([
        [24, 40], [28, 38], [32, 38], [36, 38], [40, 40], [40, 44],
        [36, 47], [32, 48], [28, 47], [24, 44], [27, 41], [32, 40],
    ], dtype=np.float32)
    landmarks[MOUTH_OUTER_LANDMARK_INDICES] = mouth_outer
    landmarks[MOUTH_INNER_LANDMARK_INDICES] = np.array([
        [27, 41], [30, 40], [34, 40], [37, 41], [37, 44], [34, 45], [30, 45], [27, 44],
    ], dtype=np.float32)
    return landmarks


class YawMaskTests(unittest.TestCase):
    def test_environment_configuration_is_read(self):
        environment = {
            "LATENTSYNC_MOUTH_YAW_THRESHOLD": "0.2",
            "LATENTSYNC_MOUTH_MAX_HORIZONTAL_SHRINK": "0.3",
            "LATENTSYNC_MOUTH_MAX_HORIZONTAL_SHIFT": "0.15",
        }
        with patch.dict("os.environ", environment, clear=True):
            self.assertEqual(yaw_mask_config(), (0.2, 0.3, 0.15, 0.02))

    def test_invalid_environment_configuration_warns_and_falls_back(self):
        environment = {
            "LATENTSYNC_MOUTH_YAW_THRESHOLD": "invalid",
            "LATENTSYNC_MOUTH_MAX_HORIZONTAL_SHRINK": "2",
            "LATENTSYNC_MOUTH_MAX_HORIZONTAL_SHIFT": "nan",
            "LATENTSYNC_MOUTH_CONTOUR_FEATHER": "-1",
        }
        with patch.dict("os.environ", environment, clear=True), self.assertWarns(RuntimeWarning):
            self.assertEqual(yaw_mask_config(), (0.12, 0.35, 0.10, 0.02))

    def test_expanded_mouth_roi_is_the_production_default(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(mouth_roi_mode(), "expanded")
            config = mouth_roi_config()
            self.assertEqual(config["mode"], "expanded")
            self.assertAlmostEqual(config["horizontal_padding"], 0.24)
            self.assertAlmostEqual(config["upper_padding"], 0.28)
            self.assertAlmostEqual(config["lower_padding"], 0.45)
            self.assertAlmostEqual(config["max_coverage"], 0.12)

            tight = mouth_roi_config("tight")
            self.assertEqual(tight["mode"], "tight")
            self.assertAlmostEqual(tight["horizontal_padding"], 0.12)
            self.assertAlmostEqual(tight["upper_padding"], 0.20)
            self.assertAlmostEqual(tight["lower_padding"], 0.30)
            self.assertAlmostEqual(tight["max_coverage"], 0.08)

    def test_mouth_roi_environment_controls_override_mode_defaults(self):
        environment = {
            "LATENTSYNC_MOUTH_ROI_MODE": "expanded",
            "LATENTSYNC_MOUTH_ROI_HORIZONTAL_PADDING": "0.23",
            "LATENTSYNC_MOUTH_ROI_UPPER_PADDING": "0.27",
            "LATENTSYNC_MOUTH_ROI_LOWER_PADDING": "0.42",
            "LATENTSYNC_MOUTH_ROI_MAX_COVERAGE": "0.13",
        }
        with patch.dict("os.environ", environment, clear=True):
            config = mouth_roi_config()
        self.assertAlmostEqual(config["horizontal_padding"], 0.23)
        self.assertAlmostEqual(config["upper_padding"], 0.27)
        self.assertAlmostEqual(config["lower_padding"], 0.42)
        self.assertAlmostEqual(config["max_coverage"], 0.13)

    def test_expanded_mouth_roi_hits_contextual_coverage_target(self):
        with patch.dict("os.environ", {}, clear=True):
            geometry = mouth_roi_geometry(synthetic_contour_landmarks(), 64, 64)
        self.assertTrue(geometry["valid"], geometry["reason"])
        self.assertGreaterEqual(geometry["feathered_coverage"], 0.10)
        self.assertLessEqual(geometry["feathered_coverage"], 0.14)

    def test_landmark_yaw_is_zero_frontal_and_signed_for_profiles(self):
        frontal = estimate_yaw_from_landmarks(synthetic_landmarks(32.0))
        left = estimate_yaw_from_landmarks(synthetic_landmarks(24.0))
        right = estimate_yaw_from_landmarks(synthetic_landmarks(40.0))
        self.assertAlmostEqual(frontal, 0.0)
        self.assertLess(left, 0.0)
        self.assertGreater(right, 0.0)
        self.assertAlmostEqual(abs(left), abs(right))

    def test_frontal_face_preserves_canonical_mask(self):
        mask = canonical_test_mask()
        adapted = adapt_canonical_mask(mask, yaw=0.05, threshold=0.1, max_shrink=0.4, max_shift=0.2)
        self.assertTrue(torch.equal(adapted, mask))

    def test_30_45_60_degree_strength_is_continuous_and_increasing(self):
        mask = canonical_test_mask()
        yaws = [np.sin(np.deg2rad(angle)) for angle in (30, 45, 60)]
        strengths = [yaw_adaptation_amount(yaw, threshold=0.1) for yaw in yaws]
        coverages = [float((1.0 - adapt_canonical_mask(mask, yaw)).sum()) for yaw in yaws]
        self.assertGreater(strengths[0], 0.0)
        self.assertLess(strengths[0], strengths[1])
        self.assertLess(strengths[1], strengths[2])
        self.assertLess(strengths[2], 1.0)
        self.assertGreater(coverages[0], coverages[1])
        self.assertGreater(coverages[1], coverages[2])

    def test_left_profile_shrinks_and_shifts_left(self):
        mask = canonical_test_mask()
        adapted = adapt_canonical_mask(mask, yaw=-1.0, threshold=0.1, max_shrink=0.4, max_shift=0.15)
        self.assertLess(float((1 - adapted).sum()), float((1 - mask).sum()))
        self.assertLess(editable_centroid(adapted), editable_centroid(mask))

    def test_right_profile_shrinks_and_shifts_right(self):
        mask = canonical_test_mask()
        adapted = adapt_canonical_mask(mask, yaw=1.0, threshold=0.1, max_shrink=0.4, max_shift=0.15)
        self.assertLess(float((1 - adapted).sum()), float((1 - mask).sum()))
        self.assertGreater(editable_centroid(adapted), editable_centroid(mask))

    def test_left_and_right_profiles_are_mirror_symmetric(self):
        mask = canonical_test_mask()
        left = adapt_canonical_mask(mask, yaw=-0.8, threshold=0.1, max_shrink=0.4, max_shift=0.15)
        right = adapt_canonical_mask(mask, yaw=0.8, threshold=0.1, max_shrink=0.4, max_shift=0.15)
        self.assertTrue(torch.allclose(left, torch.flip(right, dims=(-1,)), atol=1e-6))

    def test_adapted_mask_preserves_shape_dtype_device_and_range(self):
        mask = canonical_test_mask().to(dtype=torch.float32)
        adapted = adapt_canonical_mask(mask, yaw=0.8)
        self.assertEqual(adapted.shape, mask.shape)
        self.assertEqual(adapted.dtype, mask.dtype)
        self.assertEqual(adapted.device, mask.device)
        self.assertGreaterEqual(float(adapted.min()), 0.0)
        self.assertLessEqual(float(adapted.max()), 1.0)

    def test_extreme_transform_arguments_are_clamped_and_finite(self):
        mask = canonical_test_mask()
        adapted = adapt_canonical_mask(
            mask, yaw=10.0, threshold=-2.0, max_shrink=5.0, max_shift=5.0
        )
        self.assertTrue(torch.isfinite(adapted).all())
        self.assertEqual(adapted.shape, mask.shape)
        self.assertGreaterEqual(float(adapted.min()), 0.0)
        self.assertLessEqual(float(adapted.max()), 1.0)

    def test_contour_clipping_removes_editable_pixels_outside_face(self):
        mask = torch.zeros(1, 64, 64)
        landmarks = synthetic_contour_landmarks(left=16, right=48, top=8, bottom=60)
        contour = facial_contour_mask(landmarks, 64, 64, feather=0.03)
        final_mask = adapt_canonical_mask(mask, yaw=0.0, contour_mask=contour)
        editable = 1.0 - final_mask
        self.assertEqual(float(editable[:, :, 0].max()), 0.0)
        self.assertGreater(float(editable[:, 38:49, 24:41].mean()), 0.8)
        self.assertTrue(torch.any((editable > 0.0) & (editable < 1.0)))

    def test_detector_contour_topology_is_clockwise_and_simple(self):
        landmarks = synthetic_contour_landmarks()
        points = landmarks[FACE_CONTOUR_INDICES]
        self.assertLess(signed_polygon_area(points), 0.0)
        self.assertFalse(polygon_self_intersects(points))
        self.assertEqual(validate_polygon(points, 64, 64)[0:2], (True, "ok"))

    def test_self_intersecting_polygon_is_rejected(self):
        bow_tie = np.array([[10, 10], [50, 50], [10, 50], [50, 10]], np.float32)
        self.assertTrue(polygon_self_intersects(bow_tie))
        self.assertEqual(validate_polygon(bow_tie, 64, 64)[1], "self_intersection")

    def test_mask_polarity_one_preserves_zero_edits(self):
        image = torch.ones(1, 8, 8)
        mask = torch.ones(1, 8, 8)
        mask[:, 3:5, 3:5] = 0
        masked = image * mask
        self.assertEqual(float(masked[:, 0, 0]), 1.0)
        self.assertEqual(float(masked[:, 3, 3]), 0.0)

    def test_dual_mask_batch_keeps_conditioning_canonical(self):
        from latentsync.utils.image_processor import ImageProcessor

        processor = ImageProcessor.__new__(ImageProcessor)
        processor.mask_image = canonical_test_mask()
        processor.resize = lambda image: image
        processor.normalize = lambda image: image
        batch = processor.prepare_dual_masks_and_masked_images(
            torch.zeros(2, 3, 64, 64),
            yaws=[-0.8, 0.8],
            aligned_landmarks=[synthetic_contour_landmarks()] * 2,
            frame_offset=20,
        )
        expected_conditioning = processor.mask_image.unsqueeze(0).expand_as(batch.conditioning_mask)
        self.assertTrue(torch.equal(batch.conditioning_mask, expected_conditioning))
        self.assertTrue(torch.equal(
            batch.masked_reference_pixel_values,
            batch.reference_pixel_values * batch.conditioning_mask,
        ))
        self.assertFalse(torch.equal(
            batch.composition_editable_mask[0],
            batch.composition_editable_mask[1],
        ))

    def test_composition_formula_is_post_decode_only(self):
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline

        decoded = torch.full((1, 3, 4, 4), 0.75)
        reference = torch.full((1, 3, 4, 4), -0.25)
        composition = torch.zeros(1, 1, 4, 4)
        composition[:, :, 1:3, 1:3] = 1.0
        composed = LipsyncPipeline.compose_decoded_face(decoded, reference, composition)
        self.assertTrue(torch.equal(composed[:, :, 0, 0], reference[:, :, 0, 0]))
        self.assertTrue(torch.equal(composed[:, :, 1, 1], decoded[:, :, 1, 1]))

    def test_prepare_mask_latents_rejects_composition_role(self):
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline

        with self.assertRaisesRegex(RuntimeError, "composition_editable_mask"):
            LipsyncPipeline.prepare_mask_latents(
                object(),
                None,
                None,
                64,
                64,
                torch.float32,
                "cpu",
                None,
                False,
                mask_role="composition",
            )

    def test_prepare_mask_latents_receives_only_canonical_conditioning(self):
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline

        class _LatentDist:
            def __init__(self, batch):
                self.batch = batch

            def sample(self, generator=None):
                del generator
                return torch.zeros(self.batch, 4, 8, 8)

        class _VAE:
            config = SimpleNamespace(shift_factor=0.0, scaling_factor=1.0)

            def encode(self, images):
                self.last_images = images
                return SimpleNamespace(latent_dist=_LatentDist(images.shape[0]))

        pipeline = LipsyncPipeline.__new__(LipsyncPipeline)
        pipeline.vae = _VAE()
        pipeline.vae_scale_factor = 8
        processor = SimpleNamespace(mask_image=canonical_test_mask())
        pipeline.image_processor = processor
        conditioning = processor.mask_image.unsqueeze(0).expand(2, -1, -1, -1).clone()
        masked_reference = torch.zeros(2, 3, 64, 64)
        conditioning_latents, _ = pipeline.prepare_mask_latents(
            conditioning,
            masked_reference,
            64,
            64,
            torch.float32,
            "cpu",
            None,
            False,
        )
        self.assertEqual(tuple(conditioning_latents.shape), (1, 1, 2, 8, 8))
        with self.assertRaisesRegex(AssertionError, "canonical LatentSync mask"):
            pipeline.prepare_mask_latents(
                1.0 - conditioning,
                masked_reference,
                64,
                64,
                torch.float32,
                "cpu",
                None,
                False,
            )

    def test_mouth_roi_excludes_non_mouth_neighborhood(self):
        roi = mouth_roi_mask(synthetic_contour_landmarks(), 64, 64)
        self.assertEqual(float(roi[:, :18].max()), 0.0)
        self.assertGreater(float(roi[:, 40:48, 28:37].mean()), 0.9)

    def test_outer_lip_indices_are_the_correct_jd_cyclic_walk(self):
        self.assertEqual(
            MOUTH_LANDMARK_INDICES,
            [52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55],
        )
        self.assertEqual(MOUTH_LANDMARK_INDICES, MOUTH_OUTER_LANDMARK_INDICES)
        self.assertEqual(len(set(MOUTH_LANDMARK_INDICES)), 12)

    def test_outer_lip_polygon_is_closed_and_not_a_rectangle(self):
        landmarks = synthetic_contour_landmarks()
        closed = closed_mouth_polygon(landmarks)
        self.assertEqual(tuple(closed.shape), (13, 2))
        self.assertTrue(np.array_equal(closed[0], closed[-1]))
        self.assertGreater(signed_polygon_area(closed[:-1]), 0.0)
        self.assertFalse(polygon_self_intersects(closed[:-1]))

        geometry = mouth_roi_geometry(landmarks, 64, 64)
        raw = geometry["raw_polygon"]
        x0, y0, x1, y1 = geometry["roi_bounds"]
        bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
        self.assertLess(float(raw.sum()), 0.85 * bbox_area)

    def test_mouth_roi_centroid_and_coverage_are_bounded(self):
        geometry = mouth_roi_geometry(synthetic_contour_landmarks(), 64, 64)
        self.assertTrue(geometry["valid"], geometry["reason"])
        mouth_x, mouth_y = geometry["mouth_centroid"]
        roi_x, roi_y = geometry["roi_centroid"]
        self.assertLessEqual(abs(roi_x - mouth_x), max(0.35 * geometry["mouth_width"], 2.0))
        self.assertLessEqual(abs(roi_y - mouth_y), max(0.50 * geometry["mouth_height"], 2.0))
        self.assertLessEqual(geometry["feathered_coverage"], max_mouth_roi_coverage())

    def test_invalid_geometry_fails_closed_without_rectangular_fallback(self):
        landmarks = synthetic_contour_landmarks()
        landmarks[MOUTH_LANDMARK_INDICES[3]] = landmarks[MOUTH_LANDMARK_INDICES[0]]
        geometry = mouth_roi_geometry(landmarks, 64, 64)
        self.assertFalse(geometry["valid"])
        self.assertEqual(float(geometry["mask"].sum()), 0.0)

    def test_invalid_geometry_uses_original_canonical_mask(self):
        from latentsync.utils.image_processor import ImageProcessor

        processor = ImageProcessor.__new__(ImageProcessor)
        processor.mask_image = canonical_test_mask()
        processor.resize = lambda image: image
        processor.normalize = lambda image: image
        landmarks = synthetic_contour_landmarks()
        landmarks[MOUTH_LANDMARK_INDICES[3]] = landmarks[MOUTH_LANDMARK_INDICES[0]]
        with self.assertWarns(RuntimeWarning):
            _, masked_reference, conditioning_mask, composition_mask = processor.preprocess_dual_mask_image(
                torch.zeros(3, 64, 64), yaw=0.0,
                aligned_landmarks=landmarks, frame_index=None,
            )
        self.assertTrue(torch.equal(conditioning_mask, processor.mask_image))
        self.assertTrue(torch.equal(masked_reference, torch.zeros_like(masked_reference)))
        expected = (1.0 - processor.mask_image) * facial_contour_mask(landmarks, 64, 64)
        self.assertTrue(torch.allclose(composition_mask, expected, atol=1e-6))

    def test_frontal_three_quarter_and_profile_mouth_geometry(self):
        base = synthetic_contour_landmarks()
        for scale in (1.0, 0.65, 0.30):
            landmarks = base.copy()
            points = landmarks[MOUTH_LANDMARK_INDICES].copy()
            points[:, 0] = 32.0 + (points[:, 0] - 32.0) * scale
            landmarks[MOUTH_LANDMARK_INDICES] = points
            geometry = mouth_roi_geometry(landmarks, 64, 64)
            self.assertTrue(geometry["valid"], (scale, geometry["reason"]))
            self.assertGreater(float(geometry["mask"].sum()), 0.0)

    def test_nearby_hand_geometry_does_not_expand_mouth_roi(self):
        landmarks = synthetic_contour_landmarks()
        baseline = mouth_roi_mask(landmarks, 64, 64)
        protected = set(MOUTH_LANDMARK_INDICES) | set(FACE_CONTOUR_INDICES)
        hand_indices = [index for index in range(106) if index not in protected]
        landmarks[hand_indices] = np.array([31.0, 43.0], dtype=np.float32)
        occluded = mouth_roi_mask(landmarks, 64, 64)
        self.assertTrue(torch.equal(baseline, occluded))

    def test_final_editable_strength_is_subset_of_all_source_strengths(self):
        from latentsync.utils.image_processor import ImageProcessor

        processor = ImageProcessor.__new__(ImageProcessor)
        processor.mask_image = canonical_test_mask()
        processor.resize = lambda image: image
        processor.normalize = lambda image: image
        landmarks = synthetic_contour_landmarks()
        _, _, conditioning_mask, composition_mask = processor.preprocess_dual_mask_image(
            torch.zeros(3, 64, 64), yaw=0.7,
            aligned_landmarks=landmarks, frame_index=None,
        )
        yaw_strength = (1.0 - adapt_canonical_mask(processor.mask_image, 0.7)).clamp(0, 1)
        mouth_strength = mouth_roi_mask(landmarks, 64, 64)
        contour_strength = facial_contour_mask(landmarks, 64, 64).clamp(0, 1)
        self.assertTrue(torch.equal(conditioning_mask, processor.mask_image))
        self.assertTrue(torch.all(composition_mask <= yaw_strength + 1e-6))
        self.assertTrue(torch.all(composition_mask <= mouth_strength + 1e-6))
        self.assertTrue(torch.all(composition_mask <= contour_strength + 1e-6))

    def test_contour_intersection_never_increases_yaw_editable_coverage(self):
        canonical = canonical_test_mask()
        for yaw in (0.0, -0.85, 0.85):  # frontal and both profile examples
            yaw_mask = adapt_canonical_mask(canonical, yaw)
            contour = facial_contour_mask(synthetic_contour_landmarks(), 64, 64)
            clipped = adapt_canonical_mask(canonical, yaw, contour_mask=contour)
            self.assertTrue(torch.all((1.0 - clipped) <= (1.0 - yaw_mask) + 1e-6))
            self.assertLessEqual(float((1.0 - clipped).mean()), float((1.0 - yaw_mask).mean()))

    def test_coverage_safety_default_is_conservative(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(max_editable_coverage(), 0.18)

    def test_coverage_safety_fallback_cannot_exceed_limit(self):
        from latentsync.utils.image_processor import ImageProcessor

        processor = ImageProcessor.__new__(ImageProcessor)
        processor.mask_image = torch.zeros(1, 64, 64)
        processor.resize = lambda image: image
        processor.normalize = lambda image: image
        with patch.dict("os.environ", {"LATENTSYNC_MOUTH_MAX_EDITABLE_COVERAGE": "0.01"}), self.assertWarns(RuntimeWarning):
            _, _, _, composition_mask = processor.preprocess_dual_mask_image(
                torch.zeros(3, 64, 64), yaw=0.0,
                aligned_landmarks=synthetic_contour_landmarks(), frame_index=0,
            )
        self.assertLessEqual(float(composition_mask.mean()), 0.01)

    def test_image_processor_uses_each_frames_corresponding_yaw(self):
        from latentsync.utils.image_processor import ImageProcessor

        processor = ImageProcessor.__new__(ImageProcessor)
        processor.mask_image = canonical_test_mask().repeat(3, 1, 1)
        processor.resize = lambda image: image
        processor.normalize = lambda image: image
        images = torch.zeros(3, 3, 64, 64)
        yaws = [-0.8, 0.0, 0.8]
        landmarks = [synthetic_contour_landmarks()] * 3
        batch = processor.prepare_dual_masks_and_masked_images(
                images, yaws=yaws, aligned_landmarks=landmarks, frame_offset=10
            )
        expected_conditioning = processor.mask_image[0:1].unsqueeze(0).expand(3, -1, -1, -1)
        self.assertTrue(torch.equal(batch.conditioning_mask, expected_conditioning))
        self.assertTrue(torch.allclose(batch.masked_reference_pixel_values, torch.zeros_like(batch.masked_reference_pixel_values)))
        self.assertFalse(torch.allclose(batch.composition_editable_mask[0], batch.composition_editable_mask[1]))
        self.assertFalse(torch.allclose(batch.composition_editable_mask[1], batch.composition_editable_mask[2]))

    def test_windowed_metadata_keeps_per_frame_correspondence(self):
        from latentsync.utils.image_processor import ImageProcessor

        processor = ImageProcessor.__new__(ImageProcessor)
        processor.mask_image = canonical_test_mask()
        processor.resize = lambda image: image
        processor.normalize = lambda image: image
        originals = [synthetic_landmarks(x) for x in (24.0, 32.0, 40.0)]
        aligned = [synthetic_contour_landmarks()] * 3
        yaws = [estimate_yaw_from_landmarks(points) for points in originals]
        batch = processor.prepare_dual_masks_and_masked_images(
            torch.zeros(2, 3, 64, 64),
            yaws=yaws[1:3],
            original_landmarks=originals[1:3],
            aligned_landmarks=aligned[1:3],
            frame_offset=101,
            metadata_required=True,
        )
        self.assertEqual(tuple(batch.conditioning_mask.shape), (2, 1, 64, 64))
        self.assertEqual(tuple(batch.composition_editable_mask.shape), (2, 1, 64, 64))
        self.assertTrue(torch.equal(batch.conditioning_mask[0], canonical_test_mask()))
        self.assertTrue(torch.equal(batch.conditioning_mask[1], canonical_test_mask()))
        self.assertGreater(float(batch.composition_editable_mask[0].sum()), 0.0)
        self.assertFalse(torch.equal(batch.composition_editable_mask[1], torch.zeros_like(batch.composition_editable_mask[1])))

    def test_production_mask_is_single_channel_even_with_rgb_source(self):
        from latentsync.utils.image_processor import ImageProcessor

        processor = ImageProcessor.__new__(ImageProcessor)
        processor.mask_image = canonical_test_mask().repeat(3, 1, 1)
        processor.resize = lambda image: image
        processor.normalize = lambda image: image
        batch = processor.prepare_dual_masks_and_masked_images(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(batch.conditioning_mask.shape), (2, 1, 64, 64))
        self.assertEqual(tuple(batch.composition_editable_mask.shape), (2, 1, 64, 64))




if __name__ == "__main__":
    unittest.main()
