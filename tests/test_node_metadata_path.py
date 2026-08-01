import ast
from pathlib import Path
import unittest


def _load_actual_node_mask_boundary():
    """Compile the production metadata boundary directly from nodes.py."""
    source_path = Path(__file__).parents[1] / "nodes.py"
    tree = ast.parse(source_path.read_text())
    method = next(
        item
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EnhancedLipsyncPipeline"
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "_prepare_node_inference_masks"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_prepare_node_inference_masks"]


class _RecordingImageProcessor:
    def __init__(self):
        self.calls = []

    def prepare_masks_and_masked_images(self, faces, **kwargs):
        self.calls.append((faces, kwargs))
        return "ref", "masked", "masks"


class NodeMetadataPathTests(unittest.TestCase):
    def setUp(self):
        self.boundary = _load_actual_node_mask_boundary()
        self.processor = _RecordingImageProcessor()
        self.pipeline = type("Pipeline", (), {"image_processor": self.processor})()

    def test_window_mapping_forwards_all_metadata(self):
        yaws = [-0.5, 0.5]
        originals = [object(), object()]
        aligned = [object(), object()]
        result = self.boundary(
            self.pipeline,
            [object(), object()],
            segment_index=2,
            window_start=16,
            segment_start=160,
            yaws=yaws,
            original_landmarks=originals,
            aligned_landmarks=aligned,
        )
        self.assertEqual(result, ("ref", "masked", "masks"))
        _, kwargs = self.processor.calls[-1]
        self.assertEqual(kwargs["frame_offset"], 176)
        self.assertIs(kwargs["yaws"], yaws)
        self.assertIs(kwargs["original_landmarks"], originals)
        self.assertIs(kwargs["aligned_landmarks"], aligned)
        self.assertTrue(kwargs["metadata_required"])

    def test_window_offset_changes_with_segment_and_window_start(self):
        self.boundary(
            self.pipeline,
            [object()],
            segment_index=0,
            window_start=3,
            segment_start=80,
            yaws=[0.0],
            original_landmarks=[object()],
            aligned_landmarks=[object()],
        )
        _, kwargs = self.processor.calls[-1]
        self.assertEqual(kwargs["frame_offset"], 83)


if __name__ == "__main__":
    unittest.main()
