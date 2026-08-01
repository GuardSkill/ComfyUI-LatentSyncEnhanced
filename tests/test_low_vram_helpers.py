import unittest

from latentsync.utils.device_utils import iter_frame_batches, resolve_decode_batch_size


class LowVramHelperTests(unittest.TestCase):
    def test_decode_batch_size_is_positive_and_defensive(self):
        self.assertEqual(resolve_decode_batch_size(None), 1)
        self.assertEqual(resolve_decode_batch_size("2"), 2)
        self.assertEqual(resolve_decode_batch_size(0), 1)
        self.assertEqual(resolve_decode_batch_size("invalid", default=3), 3)

    def test_frame_batches_cover_frames_without_overlap(self):
        self.assertEqual(list(iter_frame_batches(7, 3)), [(0, 3), (3, 6), (6, 7)])
        self.assertEqual(list(iter_frame_batches(0, 3)), [])


if __name__ == "__main__":
    unittest.main()
