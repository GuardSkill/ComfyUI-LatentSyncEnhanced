# ComfyUI-LatentSyncEnhanced

An enhanced lip-sync node for ComfyUI, built on top of [ByteDance LatentSync 1.6](https://github.com/bytedance/LatentSync).

Improvements over the original wrapper:

| Feature | Original | Enhanced |
|---------|----------|----------|
| Frame with no face | ❌ Crashes | ✅ Warning + pass-through |
| All frames have no face | ❌ Crashes | ✅ Warning + return original video |
| Model path | Node's own `checkpoints/` symlink | ✅ ComfyUI standard `models/checkpoints/LatentSync-1.6/` |
| OOM on long videos | ❌ Possible | ✅ Segmenting plus bounded VAE decode and CPU offload |

---

## Prerequisites

This node is **fully self-contained** — no other custom nodes are required.
The `latentsync` inference library is bundled directly inside this package.

---

## Model Download & Storage

### Required model files

| File | Size | Purpose |
|------|------|---------|
| `latentsync_unet.pt` | ~4.8 GB | Main lip-sync UNet |
| `whisper/tiny.pt` | ~72 MB | Audio feature extractor |
| `vae/` (directory) | ~320 MB | VAE encoder/decoder |

### Where to place them

All model files go inside ComfyUI's standard **checkpoints** folder, under a subdirectory named `LatentSync-1.6`:

```
ComfyUI/
└── models/
    └── checkpoints/
        └── LatentSync-1.6/          ← create this folder
            ├── latentsync_unet.pt
            ├── whisper/
            │   └── tiny.pt
            └── vae/
                ├── config.json
                └── diffusion_pytorch_model.safetensors
```

### Download from Hugging Face

```bash
# Option 1 – huggingface-cli (recommended)
pip install huggingface_hub
huggingface-cli download ByteDance/LatentSync-1.6 \
    latentsync_unet.pt \
    whisper/tiny.pt \
    vae/config.json \
    vae/diffusion_pytorch_model.safetensors \
    --local-dir ComfyUI/models/checkpoints/LatentSync-1.6

# Option 2 – manual download
# Visit: https://huggingface.co/ByteDance/LatentSync-1.6
# Download the files above and place them as shown.
```

---

## Node Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `images` | IMAGE | — | Input video frames |
| `audio` | AUDIO | — | Input audio (any sample rate, auto-resampled to 16 kHz) |
| `seed` | INT | 1247 | Random seed for reproducibility |
| `lips_expression` | FLOAT | 1.5 | Lip movement strength (guidance scale). Range: 1.0–3.0 |
| `inference_steps` | INT | 20 | Diffusion denoising steps. More = slower but higher quality |
| `chunk_frames` | INT | 80 | Frames per processing segment. Reduce for lower CPU/RAM usage. |

### chunk_frames guide (VRAM)

| VRAM | Recommended `chunk_frames` |
|------|---------------------------|
| 24 GB | 80 (default) |
| 16 GB | 48 |
| 12 GB | 32 |
| 8 GB | 16 |

### Low-VRAM controls

The ComfyUI node interface stays unchanged. VAE decoding defaults to one
frame per CUDA call, then moves each completed batch to CPU for composition and
restoration. Advanced users can set `LATENTSYNC_DECODE_BATCH_SIZE` to a small
positive integer (for example `2`) when more VRAM is available. DeepCache is
disabled automatically on GPUs with 16 GB or less; override this with
`LATENTSYNC_ENABLE_DEEPCACHE=1`. `LATENTSYNC_MEMORY_FRACTION` is optional and
only applies when explicitly set.

For device-path diagnosis, set `LATENTSYNC_COMPOSITION_DEVICE` and/or
`LATENTSYNC_RESTORE_DEVICE` to `cpu` or `cuda`. CPU uses float32; CUDA uses the
loaded model's weight dtype. If unset, both preserve the normal low-VRAM CPU
behavior. Invalid values (and CUDA when unavailable) warn and fall back to the
normal default.

---

### Yaw-aware mouth mask controls

Near-profile mouth masks are adapted continuously from the detected landmarks.
These optional environment variables tune the adaptation; shrink and shift are
fractions of the aligned-face width:

| Variable | Default | Meaning |
|----------|---------|---------|
| `LATENTSYNC_MOUTH_YAW_THRESHOLD` | `0.12` | Normalized yaw below which the canonical mask is unchanged |
| `LATENTSYNC_MOUTH_MAX_HORIZONTAL_SHRINK` | `0.35` | Maximum horizontal shrink fraction |
| `LATENTSYNC_MOUTH_MAX_HORIZONTAL_SHIFT` | `0.10` | Maximum horizontal shift fraction |
| `LATENTSYNC_MOUTH_CONTOUR_FEATHER` | `0.02` | Face-contour feather width as a fraction of crop size |
| `LATENTSYNC_MOUTH_ROI_MODE` | `expanded` | Production `expanded` contextual polygon, or `tight` debug/ablation polygon |
| `LATENTSYNC_MOUTH_ROI_HORIZONTAL_PADDING` | `0.24` | Per-side horizontal padding as a fraction of outer-lip width |
| `LATENTSYNC_MOUTH_ROI_UPPER_PADDING` | `0.28` | Upper contextual padding as a fraction of outer-lip height |
| `LATENTSYNC_MOUTH_ROI_LOWER_PADDING` | `0.45` | Lower contextual padding as a fraction of outer-lip height |
| `LATENTSYNC_MOUTH_DILATION_FRACTION` | `0.06` | Bounded elliptical dilation radius as a fraction of mouth size |
| `LATENTSYNC_MOUTH_FEATHER_FRACTION` | `0.08` | Gaussian feather radius as a fraction of mouth size |
| `LATENTSYNC_MOUTH_ROI_MAX_COVERAGE` | `0.12` | Maximum valid landmark-derived ROI fraction of the aligned crop |

Invalid or out-of-range values emit a warning and use the documented default.
The production default is the expanded polygon: it uses the cyclic JD/InsightFace outer-lip indices
`[52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55]`, then applies bounded
morphological dilation and Gaussian feathering. Its default padding is 0.24
horizontally, 0.28 above, and 0.45 below, targeting roughly 10--14% editable
coverage. The final editable strength is intersected with the yaw-adapted mask
and facial-contour mask, so the contextual expansion cannot reach the eyes,
chin, hands, sleeves, or background outside the detected face. Invalid mouth
geometry uses the original canonical mouth mask. The diffusion polarity is
`0 = editable` and `1 = preserved`.

### Dual-mask architecture

The production path deliberately keeps three masks separate:

| Mask | Polarity | Used at |
|------|----------|---------|
| conditioning_mask | `1 = preserved`, `0 = editable` | Original canonical LatentSync mask passed to prepare_mask_latents(), masked-image VAE preparation, and UNet conditioning |
| composition_editable_mask | `1 = decoded pixel`, `0 = reference pixel` | Post-VAE composition only; may use yaw, facial contour, and tight/expanded polygon ROI |
| restoration_blend_mask | Restoration geometry | Inverse-affine face restoration only |

The composition mask is never used as a UNet conditioning mask. The production
default is canonical conditioning plus the expanded yaw-aware,
contour-clipped mouth composition mask.
Set `LATENTSYNC_MOUTH_ROI_MODE=tight` only for debug or ablation comparisons;
it is not the production default. The older padding names
`LATENTSYNC_MOUTH_HORIZONTAL_PADDING`, `LATENTSYNC_MOUTH_UPPER_PADDING`,
`LATENTSYNC_MOUTH_LOWER_PADDING`, and `LATENTSYNC_MOUTH_MAX_ROI_COVERAGE`
remain compatibility aliases, but the ROI-prefixed names take precedence.


---

## Node Outputs

| Output | Type | Description |
|--------|------|-------------|
| `images` | IMAGE | Lip-synced video frames |
| `audio` | AUDIO | Input audio resampled to 16 kHz |

---

## How no-face frames are handled

- **Some frames have no face**: those frames are passed through unchanged (original pixels). Surrounding frames' face-alignment data is used as nearest-neighbour fallback for the diffusion input so the audio sequence stays intact.
- **All frames have no face**: a warning is printed and the original video is returned as-is (no lip-sync applied, no crash).

---

## Video length vs audio length

Handled automatically inside the pipeline:

| Situation | Behaviour |
|-----------|-----------|
| Audio longer than video | Video is **ping-pong looped** to match audio length |
| Video longer than audio | Video is **trimmed** to audio length |

---

## License

Apache 2.0 — same as the underlying LatentSync model.
