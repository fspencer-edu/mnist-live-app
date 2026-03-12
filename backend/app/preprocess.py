from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps

def _to_grayscale(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    return image

def _invert(image: Image.Image) -> Image.Image:
    arr = np.array(image)
    mean_value = arr.mean()
    
    if mean_value > 127:
        image = ImageOps.invert(image)
    return image

def _threshold(image: Image.Image, threshold: int =30) -> np.ndarray:
    arr = np.array(image)
    arr = np.where(arr > threshold, arr, 0)
    return arr.astype(np.uint8)

def _bounding_box(arr: np.ndarray) -> Tuple[int, int, int, int]:
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=1)
    
    if not rows.any() or not cols.any():
        return 0, 0, arr.shape[1], arr.shape[0]
    
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    
    return left, top, right + 1, bottom + 1

def _crop_to_digit(arr: np.ndarray) -> np.ndarray:
    left, top, right, bottom = _bounding_box(arr)
    return arr[top:bottom, left:right]

def _resize_and_pad(arr: np.ndarray, target_size: int = 28, inner_size: int = 20) -> np.ndarray:
    image = Image.fromarray(arr)
    
    width, height = image.size
    if width == 0 or height == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)
    
    scale = min(inner_size / width, inner_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    canvas = Image.new("L", (target_size, target_size), color=0)
    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2
    canvas.paste(image, (left, top))
    
    return np.array(canvas)

def _center_by_mass(arr: np.ndarray) -> np.ndarray:
    """
    Shift the digit so its center of mass is near the image center.
    """
    coords = np.argwhere(arr > 0)
    if len(coords) == 0:
        return arr

    y_coords = coords[:, 0]
    x_coords = coords[:, 1]
    values = arr[y_coords, x_coords].astype(np.float32)

    total = values.sum()
    if total == 0:
        return arr

    cy = int(round((y_coords * values).sum() / total))
    cx = int(round((x_coords * values).sum() / total))

    target_y = arr.shape[0] // 2
    target_x = arr.shape[1] // 2

    shift_y = target_y - cy
    shift_x = target_x - cx

    shifted = np.zeros_like(arr)

    src_y_start = max(0, -shift_y)
    src_y_end = min(arr.shape[0], arr.shape[0] - shift_y)
    src_x_start = max(0, -shift_x)
    src_x_end = min(arr.shape[1], arr.shape[1] - shift_x)

    dst_y_start = max(0, shift_y)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_start = max(0, shift_x)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)

    shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        arr[src_y_start:src_y_end, src_x_start:src_x_end]

    return shifted

def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Convert a user image into a model-ready tensor:
    shape = (1, 28, 28, 1), dtype float32, values in [0,1]
    """
    image = _to_grayscale(image_bytes)
    image = _invert(image)

    arr = _threshold(image, threshold=30)
    arr = _crop_to_digit(arr)
    arr = _resize_and_pad(arr, target_size=28, inner_size=20)
    arr = _center_by_mass(arr)

    arr = arr.astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)   # (28, 28, 1)
    arr = np.expand_dims(arr, axis=0)    # (1, 28, 28, 1)
    return arr