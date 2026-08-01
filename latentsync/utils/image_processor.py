# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from latentsync.utils.util import read_video, write_video
from torchvision import transforms
import cv2
from einops import rearrange
import torch
import numpy as np
from typing import Optional, Union
from dataclasses import dataclass
import warnings
from .affine_transform import AlignRestore
from .face_detector import FaceDetector
from .yaw_mask import (
    adapt_canonical_mask,
    estimate_yaw_from_landmarks,
    facial_contour_mask,
    max_editable_coverage,
    mouth_roi_geometry,
    mouth_roi_config,
    transform_landmarks_to_aligned,
)



def _validate_frame_metadata(frame_index, yaw, aligned_landmarks):
    if frame_index is None:
        raise RuntimeError("Per-frame metadata is missing frame_index")
    if yaw is None or not np.isfinite(yaw):
        raise RuntimeError(f"Per-frame metadata has invalid yaw for frame {frame_index}")
    landmarks = np.asarray(aligned_landmarks) if aligned_landmarks is not None else None
    if landmarks is None or landmarks.shape != (106, 2) or not np.isfinite(landmarks).all():
        raise RuntimeError(
            f"Per-frame metadata has invalid aligned landmarks for frame {frame_index}"
        )


def load_fixed_mask(resolution: int, mask_image_path="latentsync/utils/mask.png") -> torch.Tensor:
    mask_image = cv2.imread(mask_image_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.resize(mask_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4) / 255.0
    mask_image = rearrange(torch.from_numpy(mask_image), "h w c -> c h w")
    # The PNG is RGB, but its channels encode one scalar production mask.
    return mask_image[0:1]


@dataclass(frozen=True)
class DualMaskBatch:
    """The three mask contracts carried through one aligned-face batch.

    ``conditioning_mask`` uses LatentSync's original diffusion polarity:
    ``1=preserved`` and ``0=editable``.  ``composition_editable_mask`` is a
    post-VAE polarity: ``1=use decoded pixel`` and ``0=use reference pixel``.
    The restoration blend mask is owned by ``AlignRestore`` and is deliberately
    not part of this batch.
    """

    reference_pixel_values: torch.Tensor
    masked_reference_pixel_values: torch.Tensor
    conditioning_mask: torch.Tensor
    composition_editable_mask: torch.Tensor




class ImageProcessor:
    def __init__(
        self,
        resolution: int = 512,
        device: str = "cpu",
        mask_image=None,
        restore_device: Optional[str] = None,
        restore_dtype: Optional[torch.dtype] = None,
        detector_device: Optional[str] = None,
    ):
        self.resolution = resolution
        self.device = torch.device(device)
        self.restore_device = torch.device(restore_device or device)
        self.restore_dtype = (
            torch.float32
            if self.restore_device.type == "cpu"
            else (restore_dtype or torch.float16)
        )
        self.detector_device = torch.device(detector_device or device)
        self.resize = transforms.Resize(
            (resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)

        self.restorer = AlignRestore(
            resolution=resolution,
            device=self.restore_device,
            dtype=self.restore_dtype,
        )

        if mask_image is None:
            self.mask_image = load_fixed_mask(resolution)
        else:
            self.mask_image = mask_image
        # Geometry and mask preparation are CPU-owned.  The mask is copied to
        # the diffusion device only by the VAE/UNet preparation helpers.
        self.mask_image = self.mask_image.to(device="cpu", dtype=torch.float32)
        if self.mask_image.ndim == 2:
            self.mask_image = self.mask_image.unsqueeze(0)
        if self.mask_image.ndim != 3 or self.mask_image.shape[0] not in (1, 3):
            raise ValueError(f"Canonical mask must be [1,H,W] (or RGB source), got {tuple(self.mask_image.shape)}")
        if self.mask_image.shape[0] == 3:
            self.mask_image = self.mask_image[0:1]

        if self.detector_device.type == "cpu":
            self.face_detector = None
        else:
            self.face_detector = FaceDetector(device=str(self.detector_device))


    def affine_transform(self, image: torch.Tensor, return_mask_geometry=False, frame_index=None) -> np.ndarray:
        if self.face_detector is None:
            raise NotImplementedError("Using the CPU for face detection is not supported")
        bbox, landmark_2d_106 = self.face_detector(image)
        if bbox is None:
            raise RuntimeError("Face not detected")
        detector_landmarks = np.asarray(landmark_2d_106, dtype=np.float32).copy()

        pt_left_eye = np.mean(detector_landmarks[[43, 48, 49, 51, 50]], axis=0)  # left eyebrow center
        pt_right_eye = np.mean(detector_landmarks[101:106], axis=0)  # right eyebrow center
        pt_nose = np.mean(detector_landmarks[[74, 77, 83, 86]], axis=0)  # nose center
        yaw = estimate_yaw_from_landmarks(detector_landmarks)

        landmarks3 = np.round([pt_left_eye, pt_right_eye, pt_nose])

        face, affine_matrix = self.restorer.align_warp_face(image.copy(), landmarks3=landmarks3, smooth=True)
        box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        face = rearrange(torch.from_numpy(face), "h w c -> c h w")
        if return_mask_geometry:
            aligned_landmarks = transform_landmarks_to_aligned(
                detector_landmarks,
                affine_matrix,
                self.restorer.face_size,
                self.resolution,
            )
            return (
                face,
                box,
                affine_matrix,
                yaw,
                detector_landmarks,
                aligned_landmarks,
            )
        return face, box, affine_matrix

    def preprocess_dual_mask_image(
        self,
        image: torch.Tensor,
        affine_transform=False,
        yaw=0.0,
        aligned_landmarks=None,
        frame_index=None,
        metadata_required=False,
    ):
        if metadata_required:
            metadata = {"frame_index": frame_index, "yaw": yaw,
                        "aligned_landmarks": aligned_landmarks}
            if any(value is None for value in metadata.values()):
                raise ValueError(f"Missing detected-face metadata for frame {frame_index}: {sorted(metadata)}")
            if not np.isfinite(yaw):
                raise RuntimeError(f"Detected-face mask metadata has invalid yaw for frame {frame_index}")
            landmarks = np.asarray(aligned_landmarks) if aligned_landmarks is not None else None
            if landmarks is None or landmarks.shape != (106, 2) or not np.isfinite(landmarks).all():
                raise RuntimeError(f"Detected-face mask metadata has invalid aligned landmarks for frame {frame_index}")
        if affine_transform:
            image, _, _ = self.affine_transform(image)
        else:
            image = self.resize(image)
        pixel_values = self.normalize(image / 255.0)
        canonical_mask = self.mask_image.to(device=pixel_values.device, dtype=pixel_values.dtype)
        if canonical_mask.ndim == 2:
            canonical_mask = canonical_mask.unsqueeze(0)
        if canonical_mask.ndim != 3 or canonical_mask.shape[0] not in (1, 3):
            raise RuntimeError(f"Production mask source must be CHW with 1 or 3 channels, got {tuple(canonical_mask.shape)}")
        canonical_mask = canonical_mask[0:1]
        roi_config = mouth_roi_config()
        # This tensor is the unchanged canonical LatentSync conditioning
        # contract.  Everything below that uses landmarks builds a separate
        # post-decode composition mask and must not mutate this tensor.
        conditioning_mask = canonical_mask.clone()
        yaw_adapted_mask = adapt_canonical_mask(canonical_mask, yaw)
        if aligned_landmarks is None:
            contour_mask = torch.ones_like(canonical_mask[0:1])
        else:
            contour_mask = facial_contour_mask(
                aligned_landmarks,
                canonical_mask.shape[-2],
                canonical_mask.shape[-1],
                device=canonical_mask.device,
                dtype=canonical_mask.dtype,
            )
        mouth_geometry = None
        if aligned_landmarks is None:
            local_mouth_mask = torch.ones_like(canonical_mask[0:1])
        else:
            mouth_geometry = mouth_roi_geometry(
                aligned_landmarks,
                canonical_mask.shape[-2],
                canonical_mask.shape[-1],
                mode=roi_config["mode"],
            )
            if mouth_geometry["valid"]:
                local_mouth_mask = torch.from_numpy(mouth_geometry["mask"]).to(
                    device=canonical_mask.device, dtype=canonical_mask.dtype
                ).unsqueeze(0)
            else:
                # Geometry failures use the original canonical editable mouth
                # mask explicitly.  They never become an image-sized or
                # landmark-bounding rectangle fallback.
                local_mouth_mask = (1.0 - canonical_mask).clamp(0.0, 1.0)
                warnings.warn(
                    "Invalid mouth geometry "
                    f"({mouth_geometry['reason']}); using the original canonical mouth mask",
                    RuntimeWarning,
                )
        # Mask polarity is explicit: production/diffusion mask 1=preserved,
        # 0=editable.  Geometry masks use 1=allowed editable area.  Therefore
        # the final editable strength is their intersection, never a union.
        yaw_editable = (1.0 - yaw_adapted_mask).clamp(0.0, 1.0)
        mouth_roi_strength = local_mouth_mask.clamp(0.0, 1.0)
        facial_contour_strength = contour_mask.clamp(0.0, 1.0)
        intersection = yaw_editable * mouth_roi_strength * facial_contour_strength
        # Keep the invariant explicit even if a future geometry implementation
        # changes its soft-intersection operation.
        final_editable_strength = torch.minimum(intersection, yaw_editable)
        final_editable_strength = torch.minimum(final_editable_strength, mouth_roi_strength)
        final_editable_strength = torch.minimum(final_editable_strength, facial_contour_strength)
        coverage_limit = max_editable_coverage()
        intersection_coverage = float(final_editable_strength.mean())
        final_coverage = intersection_coverage
        if final_coverage > coverage_limit:
            # Scale the already-intersected strength in place.  Every pixel
            # remains a subset of all three source strengths; no safety path
            # may reintroduce a yaw-only or contour-free edit region.
            safe_coverage_limit = float(coverage_limit) * (1.0 - 1e-6)
            final_editable_strength = final_editable_strength * (
                safe_coverage_limit / max(final_coverage, 1e-8)
            )
            warnings.warn(
                f"Final editable coverage {final_coverage:.3%} exceeds {coverage_limit:.3%}; "
                "scaling the intersected mouth strength for safety",
                RuntimeWarning,
            )
        # Required pixel-wise subset invariant:
        # final_editable_strength <= yaw_editable_strength,
        # final_editable_strength <= mouth_roi_strength, and
        # final_editable_strength <= facial_contour_strength.
        if (
            torch.any(final_editable_strength > yaw_editable + 1e-6)
            or torch.any(final_editable_strength > mouth_roi_strength + 1e-6)
            or torch.any(final_editable_strength > facial_contour_strength + 1e-6)
        ):
            raise RuntimeError("Final editable strength violated a source-mask subset invariant")
        composition_editable_mask = final_editable_strength.clamp(0.0, 1.0)
        if (
            conditioning_mask.shape != canonical_mask.shape
            or conditioning_mask.dtype != canonical_mask.dtype
            or conditioning_mask.device != canonical_mask.device
        ):
            raise RuntimeError(
                "Conditioning mask is incompatible with the canonical mask: "
                f"canonical={canonical_mask.shape}/{canonical_mask.dtype}/{canonical_mask.device}, "
                f"conditioning={conditioning_mask.shape}/{conditioning_mask.dtype}/{conditioning_mask.device}"
            )
        if not torch.isfinite(conditioning_mask).all() or conditioning_mask.min() < 0 or conditioning_mask.max() > 1:
            raise RuntimeError("Canonical conditioning mask values must be finite and within [0, 1]")
        if not torch.isfinite(composition_editable_mask).all() or composition_editable_mask.min() < 0 or composition_editable_mask.max() > 1:
            raise RuntimeError("Composition editable mask values must be finite and within [0, 1]")
        masked_pixel_values = pixel_values * conditioning_mask
        return pixel_values, masked_pixel_values, conditioning_mask[0:1], composition_editable_mask[0:1]

    def preprocess_fixed_mask_image(
        self,
        image: torch.Tensor,
        affine_transform=False,
        yaw=0.0,
        aligned_landmarks=None,
        frame_index=None,
        metadata_required=False,
    ):
        """Compatibility wrapper returning the canonical conditioning triple."""

        reference, masked_reference, conditioning_mask, _ = self.preprocess_dual_mask_image(
            image,
            affine_transform=affine_transform,
            yaw=yaw,
            aligned_landmarks=aligned_landmarks,
            frame_index=frame_index,
            metadata_required=metadata_required,
        )
        return reference, masked_reference, conditioning_mask

    def prepare_dual_masks_and_masked_images(
        self,
        images: Union[torch.Tensor, np.ndarray],
        affine_transform=False,
        yaws=None,
        aligned_landmarks=None,
        frame_offset=None,
        original_landmarks=None,
        metadata_required=False,
    ):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        production_metadata = {
            "frame_index": frame_offset, "yaw": yaws,
            "aligned_landmarks": aligned_landmarks,
            "original_landmarks": original_landmarks,
        }
        if metadata_required and any(value is None for value in production_metadata.values()):
            raise ValueError(f"Missing detected-face batch metadata: {sorted(production_metadata)}")
        if yaws is None:
            yaws = [0.0] * len(images)
        if len(yaws) != len(images):
            raise ValueError("The number of yaw values must match the number of images")
        if metadata_required and aligned_landmarks is None:
            raise RuntimeError("Detected-face batch metadata is missing aligned landmarks")
        if aligned_landmarks is None:
            aligned_landmarks = [None] * len(images)
        if len(aligned_landmarks) != len(images):
            raise ValueError("The number of landmark sets must match the number of images")
        if metadata_required:
            if original_landmarks is None or len(original_landmarks) != len(images):
                raise RuntimeError("Detected-face batch metadata is missing or misaligned original landmarks")
            for index, (yaw, original, aligned) in enumerate(zip(yaws, original_landmarks, aligned_landmarks)):
                frame_index = frame_offset + index
                original, aligned = np.asarray(original), np.asarray(aligned)
                if original.shape != (106, 2) or not np.isfinite(original).all():
                    raise RuntimeError(f"Invalid detector-space landmarks for frame {frame_index}")
                if aligned.shape != (106, 2) or not np.isfinite(aligned).all():
                    raise RuntimeError(f"Invalid aligned landmarks for frame {frame_index}")
                if yaw is None or not np.isfinite(yaw):
                    raise RuntimeError(f"Invalid signed yaw for frame {frame_index}")
        results = [
            self.preprocess_dual_mask_image(
                image,
                affine_transform=affine_transform,
                yaw=yaw,
                aligned_landmarks=landmarks,
                frame_index=None if frame_offset is None else frame_offset + index,
                metadata_required=metadata_required,
            )
            for index, (image, yaw, landmarks) in enumerate(zip(images, yaws, aligned_landmarks))
        ]

        (
            pixel_values_list,
            masked_pixel_values_list,
            conditioning_masks_list,
            composition_masks_list,
        ) = list(zip(*results))
        conditioning_mask = torch.stack(conditioning_masks_list)
        composition_editable_mask = torch.stack(composition_masks_list)
        if conditioning_mask.ndim != 4 or conditioning_mask.shape[1] != 1:
            raise RuntimeError(
                f"Conditioning mask must be [F,1,H,W], got {tuple(conditioning_mask.shape)}"
            )
        if composition_editable_mask.ndim != 4 or composition_editable_mask.shape[1] != 1:
            raise RuntimeError(
                "Composition editable mask must be [F,1,H,W], "
                f"got {tuple(composition_editable_mask.shape)}"
            )
        pixel_values = torch.stack(pixel_values_list)
        masked_pixel_values = torch.stack(masked_pixel_values_list)
        return DualMaskBatch(
            reference_pixel_values=pixel_values,
            masked_reference_pixel_values=masked_pixel_values,
            conditioning_mask=conditioning_mask,
            composition_editable_mask=composition_editable_mask,
        )

    def prepare_masks_and_masked_images(
        self,
        images: Union[torch.Tensor, np.ndarray],
        affine_transform=False,
        yaws=None,
        aligned_landmarks=None,
        frame_offset=None,
        original_landmarks=None,
        metadata_required=False,
    ):
        """Compatibility wrapper returning reference, masked reference, canonical mask."""

        batch = self.prepare_dual_masks_and_masked_images(
            images,
            affine_transform=affine_transform,
            yaws=yaws,
            aligned_landmarks=aligned_landmarks,
            frame_offset=frame_offset,
            original_landmarks=original_landmarks,
            metadata_required=metadata_required,
        )
        return (
            batch.reference_pixel_values,
            batch.masked_reference_pixel_values,
            batch.conditioning_mask,
        )

    def process_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        return pixel_values


class VideoProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu"):
        self.image_processor = ImageProcessor(resolution, device)

    def affine_transform_video(self, video_path):
        video_frames = read_video(video_path, change_fps=False)
        results = []
        for frame in video_frames:
            frame, _, _ = self.image_processor.affine_transform(frame)
            results.append(frame)
        results = torch.stack(results)

        results = rearrange(results, "f c h w -> f h w c").numpy()
        return results


if __name__ == "__main__":
    video_processor = VideoProcessor(256, "cuda")
    video_frames = video_processor.affine_transform_video("assets/demo2_video.mp4")
    write_video("output.mp4", video_frames, fps=25)
