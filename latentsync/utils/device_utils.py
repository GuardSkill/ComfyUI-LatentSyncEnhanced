"""Small helpers for keeping inference device boundaries explicit.

These functions intentionally do not import any model code.  That makes the
batching and device-policy pieces easy to test without loading CUDA, Diffusers,
or ComfyUI.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple, Union


def resolve_decode_batch_size(value: Optional[Union[int, str]], default: int = 1) -> int:
    """Return a safe positive decode batch size.

    Invalid or missing configuration falls back to ``default``.  The helper is
    used by both the pipeline and environment-variable configuration so a bad
    setting cannot create a zero-sized range or an accidental full-clip decode.
    """

    try:
        batch_size = int(value) if value is not None else int(default)
    except (TypeError, ValueError):
        batch_size = int(default)
    return max(1, batch_size)


def iter_frame_batches(length: int, batch_size: int) -> Iterator[Tuple[int, int]]:
    """Yield half-open frame ranges with a bounded batch size."""

    if length < 0:
        raise ValueError("length must be non-negative")
    batch_size = resolve_decode_batch_size(batch_size)
    for start in range(0, length, batch_size):
        yield start, min(start + batch_size, length)
