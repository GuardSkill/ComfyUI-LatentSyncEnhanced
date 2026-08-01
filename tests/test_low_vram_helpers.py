import unittest
from pathlib import Path
from unittest.mock import patch

from latentsync.utils.debug_artifacts import (
    DEBUG_DIR,
    artifact_debug_enabled,
    save_artifact_image,
)
from latentsync.utils.device_utils import (
    iter_frame_batches,
    resolve_decode_batch_size,
    resolve_processing_device,
)

class LowVramHelperTests(unittest.TestCase):
    def test_decode_batch_size_is_positive_and_defensive(self):
        self.assertEqual(resolve_decode_batch_size(None), 1)
        self.assertEqual(resolve_decode_batch_size("2"), 2)
        self.assertEqual(resolve_decode_batch_size(0), 1)
        self.assertEqual(resolve_decode_batch_size("invalid", default=3), 3)

    def test_frame_batches_cover_frames_without_overlap(self):
        self.assertEqual(list(iter_frame_batches(7, 3)), [(0, 3), (3, 6), (6, 7)])
        self.assertEqual(list(iter_frame_batches(0, 3)), [])

    def test_processing_device_preserves_default_without_override(self):
        self.assertEqual(resolve_processing_device(None, "cpu", cuda_available=True), "cpu")
        self.assertEqual(resolve_processing_device("", "cuda", cuda_available=True), "cuda")

    def test_processing_device_accepts_supported_values(self):
        self.assertEqual(resolve_processing_device(" CPU ", "cuda", cuda_available=True), "cpu")
        self.assertEqual(resolve_processing_device("cuda", "cpu", cuda_available=True), "cuda")

    def test_processing_device_warns_and_falls_back(self):
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(resolve_processing_device("mps", "cpu", cuda_available=True), "cpu")
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(resolve_processing_device("cuda", "cpu", cuda_available=False), "cpu")

    def test_artifact_debugging_is_strictly_opt_in(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(artifact_debug_enabled())
            self.assertFalse(save_artifact_image("unused", 0, object(), "zero_255"))

    def test_artifact_debugging_ignores_frames_after_limit(self):
        with patch.dict("os.environ", {"LATENTSYNC_DEBUG_ARTIFACTS": "1"}, clear=True):
            self.assertTrue(artifact_debug_enabled())
            self.assertFalse(save_artifact_image("unused", 10, object(), "zero_255"))

    def test_artifact_debug_directory_is_project_local(self):
        project_root = Path(__file__).resolve().parents[1]
        self.assertEqual(DEBUG_DIR, project_root / "debug_artifacts")


if __name__ == "__main__":
    unittest.main()
