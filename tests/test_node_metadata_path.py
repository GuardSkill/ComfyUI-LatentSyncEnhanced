import ast
import inspect
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch


class _Finite:
    def all(self):
        return True

    def __bool__(self):
        return True


class _NP:
    @staticmethod
    def asarray(value):
        return value

    @staticmethod
    def isfinite(value):
        return _Finite()


class _Landmarks:
    shape = (106, 2)


def _load_actual_node_mask_boundary():
    """Compile the production mask boundary directly from nodes.py."""
    source_path = Path(__file__).parents[1] / "nodes.py"
    tree = ast.parse(source_path.read_text())
    builder = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_build_masked_pixel_values"
    )
    method = next(
        item
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EnhancedLipsyncPipeline"
        for item in node.body
            if isinstance(item, ast.FunctionDef) and item.name == "_prepare_node_inference_masks"
    )
    module = ast.Module(body=[builder, method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "inspect": inspect,
        "np": _NP,
        "torch": torch,
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_prepare_node_inference_masks"]


def _load_actual_mask_builder():
    source_path = Path(__file__).parents[1] / "nodes.py"
    tree = ast.parse(source_path.read_text())
    builder = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_build_masked_pixel_values"
    )
    module = ast.Module(body=[builder], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_build_masked_pixel_values"]


def _load_actual_node_call_method():
    source_path = Path(__file__).parents[1] / "nodes.py"
    tree = ast.parse(source_path.read_text())
    method = next(
        item
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EnhancedLipsyncPipeline"
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "__call__"
    )
    return method


def _deleted_names(node):
    names = []
    for target in node.targets:
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                names.append(child.id)
    return names


class _RecordingImageProcessor:
    def __init__(self):
        self.kwargs = None
        self.mask_image = torch.ones(1, 4, 4)

    def prepare_dual_masks_and_masked_images(self, faces, **kwargs):
        self.kwargs = kwargs
        count = len(faces)
        conditioning = self.mask_image.unsqueeze(0).expand(count, -1, -1, -1).clone()
        composition = torch.zeros_like(conditioning)
        return SimpleNamespace(
            reference_pixel_values=torch.zeros(count, 3, 4, 4),
            masked_reference_pixel_values=torch.zeros(count, 3, 4, 4),
            conditioning_mask=conditioning,
            composition_editable_mask=composition,
        )


class NodeMetadataPathTests(unittest.TestCase):
    def setUp(self):
        self.boundary = _load_actual_node_mask_boundary()
        self.processor = _RecordingImageProcessor()
        self.pipeline = type("Pipeline", (), {"image_processor": self.processor})()

    def test_actual_node_path_preserves_window_to_global_frame_mapping(self):
        landmarks = [_Landmarks(), _Landmarks()]
        result = self.boundary(
            self.pipeline, [object(), object()], segment_index=2,
            window_start=16, segment_start=160, yaws=[-0.5, 0.5],
            original_landmarks=landmarks, aligned_landmarks=landmarks,
        )
        self.assertEqual(tuple(result.conditioning_mask.shape), (2, 1, 4, 4))
        self.assertEqual(tuple(result.composition_editable_mask.shape), (2, 1, 4, 4))
        self.assertEqual(self.processor.kwargs["frame_offset"], 176)
        self.assertTrue(self.processor.kwargs["metadata_required"])

    def test_actual_node_call_flow_has_no_second_masked_pixel_delete(self):
        """Regression guard for the cleanup path that previously hit line 1145."""
        method = _load_actual_node_call_method()
        assignments = [
            node.lineno
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "masked_pv_canonical"
                for target in node.targets
            )
        ]
        deletes = [
            (node.lineno, _deleted_names(node))
            for node in ast.walk(method)
            if isinstance(node, ast.Delete)
            and "masked_pv_canonical" in _deleted_names(node)
        ]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(len(deletes), 1)
        self.assertLess(assignments[0], deletes[0][0])
        self.assertFalse(
            any("masked_pv" in name and name != "masked_pv_canonical" for _, names in deletes for name in names)
        )


if __name__ == "__main__":
    unittest.main()
