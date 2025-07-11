#!/usr/bin/env python3
"""Render and save GIFs from saved observations.

Usage:
    python render_saved_obs_gifs.py --obs-file <path_to_observations.pt> [--output-dir <dir>] [--max-gifs 5]

The observations are expected to be saved by Buffer with `save_obs_for_rankme=True`.
This script samples consecutive frames from the saved tensor and writes animated
GIFs to disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert image array to uint8 in the range 0-255.

    Works for images already in uint8 or float images in 0-1 range.
    """
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0:
        img = img * 255.0
    return img.clip(0, 255).astype(np.uint8)


def save_gifs_from_observations(
    obs_file: Path,
    output_dir: Path | None = None,
    max_gifs: int = 5,
) -> None:
    """Load observations tensor from *obs_file* and save GIFs or images.

    For 9-channel observations (3 RGB frames stacked), generates animated GIFs.
    For 3-channel observations (single RGB frame), saves individual PNG images.
    """

    obs_file = Path(obs_file)
    if output_dir is None:
        output_dir = obs_file.parent / "gifs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading observations from {obs_file} …")
    obs: torch.Tensor = torch.load(obs_file, map_location="cpu")
    print(f"   Loaded tensor of shape {tuple(obs.shape)}, dtype={obs.dtype}")

    if obs.ndim < 3:
        raise ValueError("Observations appear to be vector data (<3 dimensions). Cannot render GIFs.")

    # Bring to numpy
    obs_np = obs.numpy()

    # Validation: expect either (N, 9, 64, 64) or (N, 3, 64, 64) format
    if obs_np.ndim != 4 or obs_np.shape[-2:] != (64, 64):
        raise ValueError(f"Expected shape (N, C, 64, 64) where C=3 or 9, got {obs_np.shape}")
    
    channels = obs_np.shape[1]
    if channels not in [3, 9]:
        raise ValueError(f"Expected 3 or 9 channels, got {channels}")
    
    is_stacked = (channels == 9)
    file_type = "GIF" if is_stacked else "PNG"
    print(f"   Detected {channels}-channel observations → will save {file_type}s")

    total_obs = obs_np.shape[0]
    print(f"   Total observations available: {total_obs:,}")

    if total_obs == 0:
        raise ValueError("No observations found in tensor")

    # Randomly select observation indices
    np.random.seed(42)  # For reproducible results
    idxs = np.random.choice(total_obs, size=min(max_gifs, total_obs), replace=False)
    idxs = sorted(idxs)  # Sort for consistent ordering in output filenames

    saved = 0
    for gif_idx, obs_idx in enumerate(idxs):
        if obs_idx >= total_obs:
            break

        obs_chw = obs_np[obs_idx]  # (C, 64, 64)
        
        if is_stacked:
            # 9-channel: Split into 3 RGB frames and create GIF
            gif_frames = []
            for f in range(3):
                frame_chw = obs_chw[f * 3 : (f + 1) * 3]  # (3, 64, 64)
                frame_hwc = np.transpose(frame_chw, (1, 2, 0))  # (64, 64, 3)
                
                # Resize to make GIF larger (3x larger = 192x192)
                frame_pil = Image.fromarray(_to_uint8(frame_hwc))
                frame_large = frame_pil.resize((192, 192), Image.NEAREST)  # 3x larger, pixelated style
                gif_frames.append(np.array(frame_large))

            output_path = output_dir / f"saved_obs_{obs_idx:05d}.gif"
            try:
                imageio.mimsave(output_path, gif_frames, duration=0.5, loop=0)
                saved += 1
                print(f"   ✅ Saved {output_path}")
            except Exception as e:
                print(f"   ⚠️ Failed to save GIF {output_path}: {e}")
        else:
            # 3-channel: Single RGB frame, save as PNG
            frame_hwc = np.transpose(obs_chw, (1, 2, 0))  # (64, 64, 3)
            
            # Resize to make image larger (3x larger = 192x192)
            frame_pil = Image.fromarray(_to_uint8(frame_hwc))
            frame_large = frame_pil.resize((192, 192), Image.NEAREST)  # 3x larger, pixelated style
            
            output_path = output_dir / f"saved_obs_{obs_idx:05d}.png"
            try:
                frame_large.save(output_path)
                saved += 1
                print(f"   ✅ Saved {output_path}")
            except Exception as e:
                print(f"   ⚠️ Failed to save PNG {output_path}: {e}")

    print(f"Done. Saved {saved} {file_type.lower()}s to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render GIFs from saved observations")
    parser.add_argument("--obs-file", type=str, default= "/scratch/tshu2/jyu197/obs_data/walker-walk/obs/observations.pt", help="Path to observations.pt file")
    parser.add_argument("--output-dir", type=str, default="/scratch/tshu2/jyu197/obs_data/walker-walk/obs", help="Directory to write GIFs")
    parser.add_argument("--max-gifs", type=int, default=10, help="Number of GIFs/images to generate")
    args = parser.parse_args()

    save_gifs_from_observations(
        Path(args.obs_file),
        Path(args.output_dir) if args.output_dir else None,
        max_gifs=args.max_gifs,
    )


if __name__ == "__main__":
    main() 