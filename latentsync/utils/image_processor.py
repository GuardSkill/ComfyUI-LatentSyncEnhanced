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
from .affine_transform import AlignRestore
from .face_detector import FaceDetector
from .yaw_mask import (
    adapt_canonical_mask,
    facial_contour_mask,
    max_editable_coverage,
    estimate_yaw_from_landmarks,
    mouth_roi_geometry,
)


def _transform_landmarks_to_aligned(
    landmarks_2d_106: np.ndarray,
    affine_matrix,
    aligned_size,
    resolution: int,
) -> np.ndarray:
    """Map detector-space landmarks through the existing aligned-face transform."""
    landmarks = np.asarray(landmarks_2d_106, dtype=np.float32)
    if isinstance(affine_matrix, torch.Tensor):
        matrix = affine_matrix.detach().cpu().numpy()
    else:
        matrix = np.asarray(affine_matrix)
    matrix = matrix.reshape(-1, 2, 3)[0].astype(np.float32)
    homogeneous = np.concatenate(
        [landmarks, np.ones((len(landmarks), 1), dtype=np.float32)], axis=1
    )
    aligned = homogeneous @ matrix.T
    crop_width, crop_height = aligned_size
    aligned[:, 0] *= resolution / float(crop_width)
    aligned[:, 1] *= resolution / float(crop_height)
    return aligned


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
    # The RGB source encodes one scalar production mask.
    return mask_image[0:1]


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

        if self.detector_device.type == "cpu":
            self.face_detector = None
        else:
            self.face_detector = FaceDetector(device=str(self.detector_device))

    def affine_transform(
        self, image: torch.Tensor, return_mask_geometry=False, frame_index=None
    ) -> np.ndarray:
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
            aligned_landmarks = _transform_landmarks_to_aligned(
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

    def preprocess_fixed_mask_image(
        self,
        image: torch.Tensor,
        affine_transform=False,
        yaw=0.0,
        aligned_landmarks=None,
        frame_index=None,
        metadata_required=False,
    ):
        if metadata_required:
            _validate_frame_metadata(frame_index, yaw, aligned_landmarks)
        if affine_transform:
            image, _, _ = self.affine_transform(image)
        else:
            image = self.resize(image)
        pixel_values = self.normalize(image / 255.0)
        canonical_mask = self.mask_image.to(
            device=pixel_values.device, dtype=pixel_values.dtype
        )
        if canonical_mask.ndim == 2:
            canonical_mask = canonical_mask.unsqueeze(0)
        if canonical_mask.ndim != 3 or canonical_mask.shape[0] not in (1, 3):
            raise ValueError(
                "Canonical mask must be CHW with one or three channels, "
                f"got {tuple(canonical_mask.shape)}"
            )
        canonical_mask = canonical_mask[0:1]
        canonical_editable = (1.0 - canonical_mask).clamp(0.0, 1.0)
        if aligned_landmarks is None:
            contour_mask = torch.ones_like(canonical_mask)
        else:
            contour_mask = facial_contour_mask(
                aligned_landmarks,
                canonical_mask.shape[-2],
                canonical_mask.shape[-1],
                device=canonical_mask.device,
                dtype=canonical_mask.dtype,
            )
        yaw_adapted_mask = adapt_canonical_mask(canonical_mask, yaw)
        mouth_geometry_fallback = False
        if aligned_landmarks is None:
            local_mouth_mask = torch.ones_like(canonical_mask)
        else:
            mouth_geometry = mouth_roi_geometry(
                aligned_landmarks,
                canonical_mask.shape[-2],
                canonical_mask.shape[-1],
            )
            if mouth_geometry["valid"]:
                local_mouth_mask = torch.from_numpy(
                    mouth_geometry["mask"]
                ).to(
                    device=canonical_mask.device,
                    dtype=canonical_mask.dtype,
                ).unsqueeze(0)
            else:
                # Invalid landmark geometry falls back to the canonical
                # editable region, never to a rectangular ROI.
                local_mouth_mask = torch.ones_like(canonical_mask)
                mouth_geometry_fallback = True

        # Geometry masks use 1 for editable/allowed pixels while the
        # production diffusion mask uses 1 for preserved pixels.
        yaw_editable = (1.0 - yaw_adapted_mask).clamp(0.0, 1.0)
        if mouth_geometry_fallback:
            yaw_editable = torch.minimum(yaw_editable, canonical_editable)
        mouth_roi_strength = local_mouth_mask.clamp(0.0, 1.0)
        facial_contour_strength = contour_mask.clamp(0.0, 1.0)
        final_editable_strength = (
            yaw_editable * mouth_roi_strength * facial_contour_strength
        )
        # Keep the pixel-wise subset invariant explicit at the final stage.
        final_editable_strength = torch.minimum(
            final_editable_strength, yaw_editable
        )
        final_editable_strength = torch.minimum(
            final_editable_strength, mouth_roi_strength
        )
        final_editable_strength = torch.minimum(
            final_editable_strength, facial_contour_strength
        )
        coverage_limit = max_editable_coverage()
        coverage = float(final_editable_strength.mean())
        if coverage > coverage_limit:
            final_editable_strength = final_editable_strength * (
                float(coverage_limit) * (1.0 - 1e-6) / coverage
            )
        mask_image = (1.0 - final_editable_strength).clamp(0.0, 1.0)
        masked_pixel_values = pixel_values * mask_image
        return pixel_values, masked_pixel_values, mask_image[0:1]

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
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")

        frame_count = len(images)
        if yaws is None:
            yaws = [0.0] * frame_count
        elif len(yaws) != frame_count:
            raise ValueError("The number of yaw values must match the number of images")
        if aligned_landmarks is None:
            aligned_landmarks = [None] * frame_count
        elif len(aligned_landmarks) != frame_count:
            raise ValueError("The number of aligned landmark sets must match the number of images")

        if metadata_required:
            if frame_offset is None:
                raise RuntimeError("Per-frame metadata is missing frame_offset")
            if original_landmarks is None or len(original_landmarks) != frame_count:
                raise RuntimeError("Per-frame metadata is missing original landmarks")
            for index, (yaw, original, aligned) in enumerate(
                zip(yaws, original_landmarks, aligned_landmarks)
            ):
                frame_index = frame_offset + index
                original = np.asarray(original) if original is not None else None
                if original is None or original.shape != (106, 2) or not np.isfinite(original).all():
                    raise RuntimeError(
                        f"Per-frame metadata has invalid original landmarks for frame {frame_index}"
                    )
                _validate_frame_metadata(frame_index, yaw, aligned)

        results = [
            self.preprocess_fixed_mask_image(
                image,
                affine_transform=affine_transform,
                yaw=yaw,
                aligned_landmarks=landmarks,
                frame_index=None if frame_offset is None else frame_offset + index,
                metadata_required=metadata_required,
            )
            for index, (image, yaw, landmarks) in enumerate(
                zip(images, yaws, aligned_landmarks)
            )
        ]

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

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
