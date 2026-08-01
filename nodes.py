"""
ComfyUI-LatentSyncEnhanced
==========================
Improvements over the original ComfyUI-LatentSyncWrapper:
  1. No crash when some (or all) frames have no face — prints warning instead.
  2. Models loaded from ComfyUI's standard checkpoints/LatentSync-1.6/ via folder_paths.
  3. Segment-based processing prevents OOM on 24 GB VRAM (default 80 frames/segment).
"""

import os
import sys
import math
import shutil
import subprocess
import tempfile
import uuid

import numpy as np
import torch
import torchaudio
import torchvision
from torchvision import transforms
import tqdm
import soundfile as sf
from omegaconf import OmegaConf
from einops import rearrange

import folder_paths



# ---------------------------------------------------------------------------
# Make this node's own latentsync package importable (self-contained)
# ---------------------------------------------------------------------------

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _NODE_DIR not in sys.path:
    sys.path.insert(0, _NODE_DIR)

from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.utils.image_processor import ImageProcessor, load_fixed_mask
from latentsync.utils.util import (
    read_video,
    read_video_file,
    read_audio,
    write_video,
    check_ffmpeg_installed,
)
from latentsync.utils.device_utils import (
    iter_frame_batches,
    resolve_decode_batch_size,
    resolve_processing_device,
)
from latentsync.utils.debug_artifacts import artifact_debug_enabled, save_artifact_image
from latentsync.models.unet import UNet3DConditionModel
from latentsync.whisper.audio2feature import Audio2Feature
from diffusers import AutoencoderKL, DDIMScheduler
from accelerate.utils import set_seed


def _build_masked_pixel_values(reference_pixel_values, conditioning_mask, *, role):
    """Construct masked-image pixels from the canonical conditioning mask."""
    del role
    assert reference_pixel_values is not None
    assert conditioning_mask is not None
    assert reference_pixel_values.ndim == 4
    assert conditioning_mask.ndim == 4
    assert reference_pixel_values.shape[0] == conditioning_mask.shape[0]
    assert conditioning_mask.shape[1] == 1
    masked_pixel_values = reference_pixel_values * conditioning_mask
    assert masked_pixel_values is not None
    assert masked_pixel_values.shape == reference_pixel_values.shape
    assert torch.isfinite(masked_pixel_values).all()
    return masked_pixel_values


# ---------------------------------------------------------------------------
# Model path resolution via ComfyUI folder_paths
# ---------------------------------------------------------------------------

def _find_latentsync_checkpoints() -> str:
    """Return the path to LatentSync-1.6 inside ComfyUI's checkpoints directories."""
    for ckpt_dir in folder_paths.get_folder_paths("checkpoints"):
        candidate = os.path.join(ckpt_dir, "LatentSync-1.6")
        if os.path.isdir(candidate):
            print(f"[LatentSyncEnhanced] Checkpoints found at: {candidate}")
            return candidate
    searched = folder_paths.get_folder_paths("checkpoints")
    raise FileNotFoundError(
        "LatentSync-1.6 not found in ComfyUI checkpoints.\n"
        "Expected: models/checkpoints/LatentSync-1.6/\n"
        f"Searched in: {searched}"
    )


# ---------------------------------------------------------------------------
# Enhanced pipeline
# ---------------------------------------------------------------------------

class EnhancedLipsyncPipeline(LipsyncPipeline):
    """
    Subclass of LipsyncPipeline with:
      - Per-segment face detection that warns instead of crashing on missing faces.
      - Segment-level processing to cap peak GPU memory usage.
    """

    # ------------------------------------------------------------------
    # Face detection helpers
    # ------------------------------------------------------------------

    def _safe_affine_transform_segment(self, video_frames, frame_offset=0):
        """
        Run face detection + affine alignment on a segment.

        Returns
        -------
        faces : torch.Tensor (N, 3, 512, 512) | None
            None when *every* frame in the segment has no face.
        boxes : list[list] | None
        matrices : list | None
        no_face_set : set[int]
            Indices (within this segment) of frames where no face was found.
        """
        faces, boxes, matrices, yaws, original_landmarks, aligned_landmarks = [], [], [], [], [], []
        no_face_indices = []

        for i, frame in enumerate(tqdm.tqdm(video_frames, desc="Face detection", leave=False)):
            frame_index = frame_offset + i
            save_artifact_image("original_frame", frame_index, frame, "zero_255")
            try:
                face, box, matrix, yaw, detector_landmarks, landmarks = self.image_processor.affine_transform(
                    frame, return_mask_geometry=True, frame_index=frame_index
                )
                save_artifact_image("aligned_face", frame_index, face, "zero_255")
                faces.append(face)
                boxes.append(box)
                matrices.append(matrix)
                yaws.append(yaw)
                original_landmarks.append(detector_landmarks)
                aligned_landmarks.append(landmarks)
            except RuntimeError as e:
                if "Face not detected" in str(e):
                    faces.append(None)
                    boxes.append(None)
                    matrices.append(None)
                    yaws.append(None)
                    original_landmarks.append(None)
                    aligned_landmarks.append(None)
                    no_face_indices.append(i)
                else:
                    raise

        total = len(video_frames)
        no_face_set = set(no_face_indices)

        if no_face_indices:
            print(
                f"[LatentSyncEnhanced] WARNING: {len(no_face_indices)}/{total} frames "
                "in segment have no detectable face — using nearest-neighbour fallback "
                "for diffusion input; those frames will be passed through unchanged."
            )

        has_face = [i for i in range(total) if i not in no_face_set]

        if not has_face:
            print(
                "[LatentSyncEnhanced] WARNING: No faces detected in this segment. "
                "Passing through original frames unchanged (no lip-sync applied)."
            )
            return None, None, None, None, None, None, no_face_set

        # Fill no-face slots with data from the nearest frame that has a face
        # (only used as diffusion input; output will still be the original frame)
        for idx in no_face_indices:
            nearest = min(has_face, key=lambda x: abs(x - idx))
            faces[idx] = faces[nearest]
            boxes[idx] = boxes[nearest]
            matrices[idx] = matrices[nearest]
            yaws[idx] = yaws[nearest]
            original_landmarks[idx] = original_landmarks[nearest]
            aligned_landmarks[idx] = aligned_landmarks[nearest]
            save_artifact_image(
                "aligned_face", frame_offset + idx, faces[idx], "zero_255"
            )
        return torch.stack(faces), boxes, matrices, yaws, original_landmarks, aligned_landmarks, no_face_set

    # ------------------------------------------------------------------
    # Restoration helper
    # ------------------------------------------------------------------

    def _restore_segment(
        self, synced_faces, original_frames, boxes, matrices, no_face_set, frame_offset=0
    ):
        """
        Paste synced faces back into original_frames.
        Frames whose index is in no_face_set are returned unchanged.
        """
        out_frames = []
        for i, face in enumerate(tqdm.tqdm(synced_faces, desc="Restoring", leave=False)):
            frame_index = frame_offset + i
            if i in no_face_set:
                out_frames.append(original_frames[i])
                if artifact_debug_enabled():
                    save_artifact_image(
                        "inverse_mask",
                        frame_index,
                        np.zeros(original_frames[i].shape[:2], dtype=np.uint8),
                        "zero_255",
                    )
                    save_artifact_image(
                        "restored_frame", frame_index, original_frames[i], "zero_255"
                    )
                continue
            x1, y1, x2, y2 = boxes[i]
            h = int(y2 - y1)
            w = int(x2 - x1)
            face = torchvision.transforms.functional.resize(
                face, size=(h, w),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
            save_artifact_image(
                "face_before_restore", frame_index, face, "minus_one_one"
            )
            out_frame = self.image_processor.restorer.restore_img(
                original_frames[i], face, matrices[i], debug_frame_index=frame_index
            )
            out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)

    # ------------------------------------------------------------------
    # Video looping (WITHOUT face detection)
    # ------------------------------------------------------------------

    def _loop_frames_only(self, video_frames: np.ndarray, target_length: int) -> np.ndarray:
        """Ping-pong loop video_frames until target_length, no face detection."""
        if len(video_frames) >= target_length:
            return video_frames[:target_length]
        num_loops = math.ceil(target_length / len(video_frames))
        parts = [
            video_frames if i % 2 == 0 else video_frames[::-1]
            for i in range(num_loops)
        ]
        return np.concatenate(parts, axis=0)[:target_length]

    # ------------------------------------------------------------------
    # Main __call__ — segment-based, OOM-safe
    # ------------------------------------------------------------------

    def _prepare_node_inference_masks(
        self, inference_faces, *, segment_index, window_start, segment_start,
        yaws, original_landmarks, aligned_landmarks,
    ):
        """Build canonical conditioning and post-decode composition masks."""
        del segment_index
        frame_indices = [segment_start + window_start + n for n in range(len(inference_faces))]
        metadata = {
            "frame_indices": frame_indices,
            "yaw_values": yaws,
            "original_landmarks": original_landmarks,
            "aligned_landmarks": aligned_landmarks,
        }
        lengths = {key: None if value is None else len(value) for key, value in metadata.items()}
        if any(value is None for value in metadata.values()) or any(
            length != len(inference_faces) for length in lengths.values()
        ):
            raise RuntimeError(f"Missing or misaligned production metadata: lengths={lengths}")
        for frame_index, yaw, original, aligned in zip(
            frame_indices, yaws, original_landmarks, aligned_landmarks
        ):
            if yaw is None or not np.isfinite(yaw):
                raise RuntimeError(f"Invalid signed yaw for frame {frame_index}")
            original = np.asarray(original)
            aligned = np.asarray(aligned)
            if original.shape != (106, 2) or not np.isfinite(original).all():
                raise RuntimeError(f"Invalid detector-space landmarks for frame {frame_index}")
            if aligned.shape != (106, 2) or not np.isfinite(aligned).all():
                raise RuntimeError(f"Invalid aligned landmarks for frame {frame_index}")
        mask_batch = self.image_processor.prepare_dual_masks_and_masked_images(
            inference_faces,
            affine_transform=False,
            yaws=yaws,
            aligned_landmarks=aligned_landmarks,
            frame_offset=segment_start + window_start,
            original_landmarks=original_landmarks,
            metadata_required=True,
        )
        ref_pv = mask_batch.reference_pixel_values
        conditioning_mask = mask_batch.conditioning_mask
        masked_pv_canonical = _build_masked_pixel_values(
            ref_pv, conditioning_mask, role="canonical"
        )
        assert masked_pv_canonical is not None
        assert masked_pv_canonical.shape == ref_pv.shape
        assert torch.isfinite(masked_pv_canonical).all()
        canonical_source = self.image_processor.mask_image.to(
            device=conditioning_mask.device, dtype=conditioning_mask.dtype
        )
        if canonical_source.ndim == 3:
            canonical_source = canonical_source.unsqueeze(0)
        if not torch.equal(conditioning_mask, canonical_source.expand_as(conditioning_mask)):
            raise RuntimeError("conditioning_mask deviated from the original canonical LatentSync mask")
        if not torch.allclose(masked_pv_canonical, mask_batch.masked_reference_pixel_values, atol=1e-4, rtol=0.0):
            raise RuntimeError("masked_reference_pixel_values were not built from conditioning_mask")
        return mask_batch

    @torch.no_grad()
    def _denoise_window(
        self,
        latents,
        conditioning_mask_latents,
        conditioning_masked_image_latents,
        reference_latents,
        audio_embeds,
        timesteps,
        num_inference_steps,
        guidance_scale,
        extra_step_kwargs,
        callback=None,
        callback_steps=1,
    ):
        """Run the unchanged denoising equations for one canonical condition."""
        do_cfg = guidance_scale > 1.0
        num_warmup = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as pbar:
            for j, timestep in enumerate(timesteps):
                unet_in = torch.cat([latents] * 2) if do_cfg else latents
                unet_in = self.scheduler.scale_model_input(unet_in, timestep)
                unet_in = torch.cat(
                    [unet_in, conditioning_mask_latents, conditioning_masked_image_latents, reference_latents],
                    dim=1,
                )
                noise_pred = self.unet(unet_in, timestep, encoder_hidden_states=audio_embeds).sample
                if do_cfg:
                    noise_uncond, noise_audio = noise_pred.chunk(2)
                    noise_pred = noise_uncond + guidance_scale * (noise_audio - noise_uncond)
                latents = self.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs).prev_sample
                if j == len(timesteps) - 1 or (
                    (j + 1) > num_warmup and (j + 1) % self.scheduler.order == 0
                ):
                    pbar.update()
                    if callback is not None and j % callback_steps == 0:
                        callback(j, timestep, latents)
        return latents

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        video_mask_path: str = None,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height=None,
        width=None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype=torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        generator=None,
        callback=None,
        callback_steps=1,
        chunk_frames: int = 80,
        decode_batch_size: int = 1,
        cpu_offload: bool = True,
        **kwargs,
    ):
        is_train = self.unet.training
        self.unet.eval()
        check_ffmpeg_installed()

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width  = width  or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(height, width, callback_steps)

        device = self._execution_device
        decode_batch_size = resolve_decode_batch_size(decode_batch_size)
        default_aux_device = "cpu" if cpu_offload else device.type
        composition_device = torch.device(resolve_processing_device(
            os.getenv("LATENTSYNC_COMPOSITION_DEVICE"),
            default_aux_device,
            cuda_available=torch.cuda.is_available(),
        ))
        restore_device = torch.device(resolve_processing_device(
            os.getenv("LATENTSYNC_RESTORE_DEVICE"),
            default_aux_device,
            cuda_available=torch.cuda.is_available(),
        ))
        composition_dtype = (
            torch.float32 if composition_device.type == "cpu" else weight_dtype
        )
        restore_dtype = torch.float32 if restore_device.type == "cpu" else weight_dtype
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(
            height,
            device=str(device),
            mask_image=mask_image,
            # Keep geometric restoration off CUDA when decoded faces are
            # streamed to CPU.  The detector may still use CUDA.
            restore_device=str(restore_device),
            restore_dtype=restore_dtype,
            detector_device=str(device),
        )
        print(
            "[LatentSyncEnhanced] Device policy: "
            f"diffusion={device}, decode_batch_size={decode_batch_size}"
        )
        print(
            "[LatentSyncEnhanced] "
            f"composition device/dtype: {composition_device}/{composition_dtype}"
        )
        print(
            "[LatentSyncEnhanced] "
            f"restoration device/dtype: {restore_device}/{restore_dtype}"
        )
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        do_cfg = guidance_scale > 1.0
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # ── Audio (processed once for the full clip) ────────────────────
        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks  = self.audio_encoder.feature2chunks(
            feature_array=whisper_feature, fps=video_fps
        )
        # Whisper is only needed while creating CPU audio chunks.  Releasing it
        # before diffusion makes the CUDA residency contract explicit.
        if cpu_offload and hasattr(self.audio_encoder, "model"):
            self.audio_encoder.model.to("cpu")
            print("[LatentSyncEnhanced] Offloaded Whisper encoder to CPU after feature extraction.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        audio_samples = read_audio(audio_path)

        # Use actual audio duration (samples) as ground truth for target frame count.
        # whisper_chunks can be longer than the real audio because Whisper pads short
        # audio to its 30-second context window before encoding.
        actual_audio_frames = int(len(audio_samples) / audio_sample_rate * video_fps)
        whisper_chunks = whisper_chunks[:actual_audio_frames]

        # ── Video (loop to match audio, no face detection yet) ──────────
        # The node already encoded the input at video_fps.  Do not silently
        # transcode it back to the legacy 25 FPS default in read_video().
        video_frames = read_video(video_path, change_fps=False, use_decord=False)
        video_frames = self._loop_frames_only(video_frames, actual_audio_frames)

        total_frames = len(video_frames)
        num_channels_latents = self.vae.config.latent_channels
        print(f"[LatentSyncEnhanced] Total frames: {total_frames} | Chunk size: {chunk_frames}")

        all_synced_frames = []

        # ── Segment loop ────────────────────────────────────────────────
        for seg_start in range(0, total_frames, chunk_frames):
            seg_end    = min(seg_start + chunk_frames, total_frames)
            seg_video  = video_frames[seg_start:seg_end]
            seg_whisper = whisper_chunks[seg_start:seg_end]

            print(
                f"[LatentSyncEnhanced] Processing segment {seg_start}–{seg_end} "
                f"({seg_end - seg_start} frames)..."
            )

            seg_faces, seg_boxes, seg_matrices, seg_yaws, seg_original_landmarks, seg_landmarks, no_face_set = \
                self._safe_affine_transform_segment(seg_video, frame_offset=seg_start)

            if seg_faces is None:
                # Every frame in this segment has no face → pass through unchanged
                all_synced_frames.append(seg_video.copy())
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            # Allocate latents ONLY for this segment (avoids O(total_frames) allocation)
            seg_latents = self.prepare_latents(
                1, len(seg_whisper), num_channels_latents,
                height, width, weight_dtype, device, generator,
            )

            synced_chunk = []
            num_inferences = math.ceil(len(seg_whisper) / num_frames)

            for i in tqdm.tqdm(range(num_inferences), desc="Diffusion", leave=False):
                # Audio embeddings for this 16-frame batch
                if self.unet.add_audio_layer:
                    audio_embeds = torch.stack(
                        seg_whisper[i * num_frames : (i + 1) * num_frames]
                    ).to(device, dtype=weight_dtype)
                    if do_cfg:
                        null = torch.zeros_like(audio_embeds)
                        audio_embeds = torch.cat([null, audio_embeds])
                else:
                    audio_embeds = None

                inference_faces = seg_faces[i * num_frames : (i + 1) * num_frames]
                inference_yaws = seg_yaws[i * num_frames : (i + 1) * num_frames]
                inference_landmarks = seg_landmarks[i * num_frames : (i + 1) * num_frames]
                inference_original_landmarks = seg_original_landmarks[i * num_frames : (i + 1) * num_frames]
                latents         = seg_latents[:, :, i * num_frames : (i + 1) * num_frames]

                window_start = i * num_frames
                mask_batch = self._prepare_node_inference_masks(
                    inference_faces,
                    segment_index=seg_start // chunk_frames,
                    window_start=window_start,
                    segment_start=seg_start,
                    yaws=inference_yaws,
                    original_landmarks=inference_original_landmarks,
                    aligned_landmarks=inference_landmarks,
                )
                ref_pv = mask_batch.reference_pixel_values
                conditioning_mask = mask_batch.conditioning_mask
                composition_editable_mask = mask_batch.composition_editable_mask
                masked_pv_canonical = _build_masked_pixel_values(
                    ref_pv,
                    conditioning_mask,
                    role="canonical",
                )
                assert masked_pv_canonical is not None
                assert masked_pv_canonical.shape == ref_pv.shape
                assert torch.isfinite(masked_pv_canonical).all()
                if not torch.allclose(
                    masked_pv_canonical,
                    mask_batch.masked_reference_pixel_values,
                    atol=1e-4,
                    rtol=0.0,
                ):
                    raise RuntimeError(
                        "canonical masked pixel values diverged from ImageProcessor output"
                    )
                assert masked_pv_canonical is not None
                assert masked_pv_canonical.shape == ref_pv.shape
                assert torch.isfinite(masked_pv_canonical).all()
                conditioning_mask_latents, conditioning_masked_image_latents = self.prepare_mask_latents(
                    conditioning_mask,
                    masked_pv_canonical,
                    height,
                    width,
                    weight_dtype,
                    device,
                    generator,
                    do_cfg,
                    mask_role="conditioning",
                )
                ref_latents = self.prepare_image_latents(
                    ref_pv, device, weight_dtype, generator, do_cfg
                )

                latents = self._denoise_window(
                    latents,
                    conditioning_mask_latents,
                    conditioning_masked_image_latents,
                    ref_latents,
                    audio_embeds,
                    timesteps,
                    num_inference_steps,
                    guidance_scale,
                    extra_step_kwargs,
                    callback=callback,
                    callback_steps=callback_steps,
                )

                # The UNet no longer needs its conditioning tensors.  Delete
                # them before entering VAE decode so their allocations cannot
                # overlap the decoder's peak activation memory.
                assert masked_pv_canonical is not None
                assert masked_pv_canonical.shape == ref_pv.shape
                assert torch.isfinite(masked_pv_canonical).all()
                del (
                    masked_pv_canonical,
                    conditioning_mask_latents,
                    conditioning_masked_image_latents,
                    ref_latents,
                )
                if audio_embeds is not None:
                    del audio_embeds
                if "unet_in" in locals():
                    del unet_in
                if "noise_pred" in locals():
                    del noise_pred
                if "noise_uncond" in locals():
                    del noise_uncond
                if "noise_audio" in locals():
                    del noise_audio
                if "null" in locals():
                    del null

                # Decode and restore one bounded batch at a time.  In CPU
                # offload mode every operation after VAE.decode is explicitly
                # CPU-owned, so no CUDA tensor is retained by synced_chunk.
                for decode_start, decode_end in iter_frame_batches(
                    latents.shape[2], decode_batch_size
                ):
                    decoded = next(
                        self.decode_latents_stream(
                            latents[:, :, decode_start:decode_end],
                            decode_batch_size=decode_batch_size,
                            output_device=composition_device,
                        )
                    )
                    decoded = decoded.to(device=composition_device, dtype=composition_dtype)
                    ref_slice = ref_pv[decode_start:decode_end]
                    composition_slice = composition_editable_mask[decode_start:decode_end]
                    if artifact_debug_enabled():
                        for batch_index in range(decoded.shape[0]):
                            frame_index = window_start + decode_start + batch_index
                            frame_index += seg_start
                            save_artifact_image(
                                "decoded_face",
                                frame_index,
                                decoded[batch_index],
                                "minus_one_one",
                            )
                    composed = self.paste_surrounding_pixels_back(
                        decoded,
                        ref_slice,
                        composition_slice,
                        composition_device,
                        composition_dtype,
                    )
                    composed = composed.to(device=restore_device, dtype=restore_dtype)

                    local_no_face_set = {
                        index - decode_start
                        for index in no_face_set
                        if decode_start <= index < decode_end
                    }
                    restored = self._restore_segment(
                        composed,
                        seg_video[window_start + decode_start : window_start + decode_end],
                        seg_boxes[window_start + decode_start : window_start + decode_end],
                        seg_matrices[window_start + decode_start : window_start + decode_end],
                        local_no_face_set,
                        frame_offset=seg_start + window_start + decode_start,
                    )
                    synced_chunk.append(restored)
                    del decoded, ref_slice, composition_slice, composed, restored

                # ref_pv and the two independent mask tensors are CPU storage
                # needed only for this
                # window; release their references before the next window.
                del (
                    ref_pv,
                    conditioning_mask,
                    composition_editable_mask,
                    mask_batch,
                    latents,
                )

            # Restore faces → numpy output for this segment
            all_synced_frames.append(np.concatenate(synced_chunk, axis=0))

            # Aggressively free segment allocations
            del seg_faces, seg_boxes, seg_matrices, seg_yaws, seg_original_landmarks, seg_landmarks, seg_latents, synced_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Assemble and write output ────────────────────────────────────
        synced_video_frames = np.concatenate(all_synced_frames, axis=0)

        audio_remain = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_np = audio_samples[:audio_remain].cpu().numpy()

        if is_train:
            self.unet.train()

        out_dir = os.path.dirname(video_out_path)
        tmp_video = os.path.join(out_dir, "_enhanced_tmp_video.mp4")
        tmp_audio = os.path.join(out_dir, "_enhanced_tmp_audio.wav")

        write_video(tmp_video, synced_video_frames, fps=video_fps)
        sf.write(tmp_audio, audio_np, audio_sample_rate)

        cmd = (
            f"ffmpeg -y -loglevel error -nostdin "
            f"-i {tmp_video} -i {tmp_audio} "
            f"-c:v libx264 -crf 18 -c:a aac -q:v 0 -q:a 0 -shortest "
            f"{video_out_path}"
        )
        subprocess.run(cmd, shell=True)

        for f in [tmp_video, tmp_audio]:
            try:
                os.remove(f)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class LatentSyncEnhancedNode:
    """
    Lip-sync node with:
    - Graceful handling of frames without a detectable face.
    - Model loading from ComfyUI's standard checkpoints/LatentSync-1.6/.
    - Configurable chunk_frames to prevent OOM on 24 GB VRAM.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":           ("IMAGE",),
                "audio":            ("AUDIO",),
                "seed":             ("INT", {"default": 1247, "min": 0, "max": 2**32 - 1}),
                "lips_expression":  ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
                "inference_steps":  ("INT",   {"default": 20,   "min": 1,   "max": 100,  "step": 1}),
                "chunk_frames":     ("INT",   {
                    "default": 80,
                    "min": 16,
                    "max": 512,
                    "step": 16,
                    "tooltip": (
                        "Frames processed per segment. "
                        "80 ≈ 3.2 s @25 fps and is safe for 24 GB VRAM. "
                        "Reduce if you still hit OOM; increase for speed."
                    ),
                }),
                "video_fps":        ("FLOAT", {
                    "default": 25.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.001,
                    "tooltip": "FPS of the input video frames. Must match the source video fps (e.g. 23.976, 24, 25, 29.97, 30).",
                }),
            }
        }

    RETURN_TYPES  = ("IMAGE", "AUDIO")
    RETURN_NAMES  = ("images", "audio")
    FUNCTION      = "inference"
    CATEGORY      = "LatentSync"

    # ------------------------------------------------------------------

    def _load_pipeline(self, ckpt_dir: str, config, weight_dtype) -> EnhancedLipsyncPipeline:
        """Instantiate and return an EnhancedLipsyncPipeline on CUDA."""
        # Scheduler — created directly (no HuggingFace download)
        scheduler = DDIMScheduler(
            beta_end=0.012,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            clip_sample=False,
            num_train_timesteps=1000,
            prediction_type="epsilon",
            set_alpha_to_one=False,
            steps_offset=1,
        )

        # Whisper audio encoder
        if config.model.cross_attention_dim == 384:
            whisper_name = "tiny"
        elif config.model.cross_attention_dim == 768:
            whisper_name = "small"
        else:
            raise ValueError(f"Unsupported cross_attention_dim: {config.model.cross_attention_dim}")

        whisper_path = os.path.join(ckpt_dir, "whisper", f"{whisper_name}.pt")
        if not os.path.exists(whisper_path):
            print(
                f"[LatentSyncEnhanced] WARNING: {whisper_path} not found, "
                f"falling back to online Whisper model '{whisper_name}'."
            )
            whisper_path = whisper_name

        audio_encoder = Audio2Feature(
            model_path=whisper_path,
            device="cuda",
            num_frames=config.data.num_frames,
            audio_feat_length=config.data.audio_feat_length,
        )

        # VAE
        vae_dir = os.path.join(ckpt_dir, "vae")
        vae = AutoencoderKL.from_pretrained(vae_dir, torch_dtype=weight_dtype, local_files_only=True)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor   = 0

        # UNet
        unet_path = os.path.join(ckpt_dir, "latentsync_unet.pt")
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            unet_path,
            device="cpu",
        )
        unet = unet.to(dtype=weight_dtype)

        pipeline = EnhancedLipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        ).to("cuda")

        # DeepCache retains UNet activations.  Keep it for larger cards, but
        # disable it automatically on low-VRAM GPUs unless explicitly forced.
        deepcache_setting = os.getenv("LATENTSYNC_ENABLE_DEEPCACHE", "auto").lower()
        low_vram = (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) <= 16
        )
        enable_deepcache = deepcache_setting == "1" or (
            deepcache_setting == "auto" and not low_vram
        )
        if enable_deepcache:
            try:
                from DeepCache import DeepCacheSDHelper
                helper = DeepCacheSDHelper(pipe=pipeline)
                helper.set_params(cache_interval=3, cache_branch_id=0)
                helper.enable()
                print("[LatentSyncEnhanced] DeepCache enabled.")
            except ImportError:
                print("[LatentSyncEnhanced] DeepCache not available, skipping.")
        else:
            print("[LatentSyncEnhanced] DeepCache disabled by low-VRAM policy.")

        return pipeline

    # ------------------------------------------------------------------

    def inference(self, images, audio, seed, lips_expression, inference_steps, chunk_frames, video_fps):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_fp16 = False

        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            use_fp16   = torch.cuda.get_device_capability()[0] > 7
            if gpu_mem_gb > 20:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            memory_fraction = os.getenv("LATENTSYNC_MEMORY_FRACTION")
            if memory_fraction:
                torch.cuda.set_per_process_memory_fraction(float(memory_fraction))
                print(
                    "[LatentSyncEnhanced] CUDA allocator fraction set to "
                    f"{memory_fraction} by LATENTSYNC_MEMORY_FRACTION."
                )

        weight_dtype = torch.float16 if use_fp16 else torch.float32

        # Temp directory (cleaned up in finally block)
        run_id   = uuid.uuid4().hex[:8]
        temp_dir = os.path.join(tempfile.gettempdir(), f"latentsync_enhanced_{run_id}")
        os.makedirs(temp_dir, exist_ok=True)

        temp_video_path  = os.path.join(temp_dir, "input.mp4")
        output_video_path = os.path.join(temp_dir, "output.mp4")
        audio_path       = os.path.join(temp_dir, "audio.wav")

        try:
            # ── Prepare input frames ────────────────────────────────────
            if isinstance(images, list):
                frames = torch.stack(images)
            else:
                frames = images
            frames_uint8 = (frames.cpu() * 255).to(torch.uint8)

            # ImageIO is the maintained video I/O boundary.  TorchVision's
            # encode_video/read_video APIs are deprecated and version-sensitive.
            write_video(temp_video_path, frames_uint8.numpy(), fps=video_fps)

            # ── Prepare input audio ─────────────────────────────────────
            waveform    = audio["waveform"]
            sample_rate = audio["sample_rate"]
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(device)
                waveform  = resampler(waveform.to(device))
                sample_rate = 16000
            torchaudio.save(audio_path, waveform.cpu(), sample_rate)
            resampled_audio = {"waveform": waveform.unsqueeze(0).cpu(), "sample_rate": sample_rate}

            # ── Load config ─────────────────────────────────────────────
            config_512 = os.path.join(_NODE_DIR, "configs", "unet", "stage2_512.yaml")
            config_256 = os.path.join(_NODE_DIR, "configs", "unet", "stage2.yaml")
            config_path = config_512 if os.path.exists(config_512) else config_256
            config = OmegaConf.load(config_path)

            mask_image_path = os.path.join(_NODE_DIR, "latentsync", "utils", "mask.png")
            if not os.path.exists(mask_image_path):
                raise FileNotFoundError(f"Mask image not found: {mask_image_path}")

            # ── Resolve checkpoint dir ──────────────────────────────────
            ckpt_dir = _find_latentsync_checkpoints()

            # ── Build pipeline ──────────────────────────────────────────
            pipeline = self._load_pipeline(ckpt_dir, config, weight_dtype)

            if seed != -1:
                set_seed(seed)
            else:
                torch.seed()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ── Run inference ───────────────────────────────────────────
            pipeline(
                video_path=temp_video_path,
                audio_path=audio_path,
                video_out_path=output_video_path,
                num_frames=config.data.num_frames,
                num_inference_steps=inference_steps,
                guidance_scale=lips_expression,
                weight_dtype=weight_dtype,
                width=config.data.resolution,
                height=config.data.resolution,
                mask_image_path=mask_image_path,
                chunk_frames=chunk_frames,
                video_fps=video_fps,
                # Keep the ComfyUI node inputs unchanged.  Advanced users can
                # tune this internal bound with LATENTSYNC_DECODE_BATCH_SIZE.
                decode_batch_size=resolve_decode_batch_size(
                    os.getenv("LATENTSYNC_DECODE_BATCH_SIZE", "1")
                ),
                cpu_offload=True,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not os.path.exists(output_video_path):
                raise FileNotFoundError(f"Output video not found: {output_video_path}")

            # ── Read result ─────────────────────────────────────────────
            processed_frames = read_video_file(output_video_path)
            processed_frames = torch.from_numpy(processed_frames).float() / 255.0

            return (processed_frames.cpu(), resampled_audio)

        except Exception as e:
            print(f"[LatentSyncEnhanced] Error during inference: {e}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            for f in [temp_video_path, output_video_path, audio_path]:
                if f and os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "LatentSyncEnhanced": LatentSyncEnhancedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentSyncEnhanced": "LatentSync Enhanced (No-Face Safe + OOM Guard)",
}
