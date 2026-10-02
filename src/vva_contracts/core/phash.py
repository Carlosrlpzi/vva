"""Perceptual hashing and near-duplicate search between dataset splits.

Why this module exists
----------------------
Consecutive frames of the same camera are almost identical. If one lands in
``train`` and its neighbour in ``val``, the model is validated on images it
has effectively seen, and mAP is inflated. Exact byte hashes miss this (a new
JPEG encoding changes every byte), so we need a *perceptual* hash: a short
fingerprint whose Hamming distance tracks visual similarity.

dHash (difference hash)
-----------------------
1. Convert to grayscale and resize to 9x8 pixels (destroys detail, keeps
   coarse structure; also normalises resolution and aspect ratio).
2. For each of the 8 rows compare each pixel with its right neighbour:
   bit = 1 if left > right. 8 rows x 8 comparisons = 64 bits.
Gradients survive recompression and small brightness changes, which is why
comparisons beat raw intensities.

Banded search (same idea as LSH banding for MinHash)
----------------------------------------------------
Comparing every train image with every val image is O(N*M). Instead split the
64 bits into 8 bands of 8 bits. Pigeonhole principle: if two hashes differ in
at most 7 bits, those differences fall in at most 7 bands, so at least one
band is *identical*. Bucketing by (band_index, band_value) therefore finds
every pair with distance <= 7 exactly, with no false negatives, and only
compares pairs that share a bucket.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from PIL import Image

N_BANDS = 8
BAND_BITS = 8
# Largest distance for which the pigeonhole guarantee above holds.
MAX_GUARANTEED_DISTANCE = N_BANDS - 1


def dhash(path: Path) -> int:
    """Return the 64-bit dHash of an image file as a Python int."""
    with Image.open(path) as img:
        # "L" = 8-bit grayscale. LANCZOS is the stable, deterministic filter
        # documented by Pillow; another filter would change the bits.
        small = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        # tobytes() on an 'L' image is one byte per pixel, row-major. Unlike
        # getdata() (deprecated in Pillow 12) it exists in every Pillow version,
        # including the older one shipped with Raspberry Pi OS.
        pixels = list(small.tobytes())
    value = 0
    for row in range(8):
        for col in range(8):
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            # Shift in one bit per comparison: the first comparison ends up as
            # the most significant bit, giving a fixed, documented bit order.
            value = (value << 1) | int(left > right)
    return value


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes (popcount of XOR)."""
    return (a ^ b).bit_count()


def _bands(value: int) -> Iterable[tuple[int, int]]:
    """Yield (band_index, band_value) for the 8 bands of a 64-bit hash."""
    mask = (1 << BAND_BITS) - 1
    for band in range(N_BANDS):
        yield band, (value >> (band * BAND_BITS)) & mask


def cross_near_duplicates(left: dict[str, int], right: dict[str, int], max_distance: int) -> list[tuple[str, str, int]]:
    """All (left_key, right_key, distance) pairs with distance <= max_distance.

    ``max_distance`` must be <= MAX_GUARANTEED_DISTANCE; above it the banded
    search could miss pairs, so the caller (the request validator) forbids it.
    """
    if max_distance > MAX_GUARANTEED_DISTANCE:
        raise ValueError(f"max_distance must be <= {MAX_GUARANTEED_DISTANCE} for an exact search")
    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for key, value in right.items():
        for band in _bands(value):
            buckets[band].append(key)
    found: dict[tuple[str, str], int] = {}
    checked: set[tuple[str, str]] = set()
    for left_key, left_value in left.items():
        for band in _bands(left_value):
            for right_key in buckets.get(band, []):
                pair = (left_key, right_key)
                if pair in checked:
                    continue  # the same pair can share several bands
                checked.add(pair)
                distance = hamming(left_value, right[right_key])
                if distance <= max_distance:
                    found[pair] = distance
    return sorted((a, b, d) for (a, b), d in found.items())
