"""Standalone component segmentation inference script."""
from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - optional dependency in the execution environment
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - exercised by the test suite
    np = None  # type: ignore

if np is not None and getattr(np, "IS_FAKE", False):  # pragma: no cover - test shim
    np = None  # type: ignore

# Lazy import for skimage to avoid numpy binary compatibility issues during test imports
skio = None  # Will be imported on first use if available

from mrcnn.config import Config

CLASS_NAMES = [
    "BG",
    "magnet",
    "fpc",
    "rw_head",
    "spindle_hub",
    "platters_clamp",
    "platter",
    "bay",
    "lid",
    "pcb",
    "head_contacts",
    "top_dumper",
]


class ComponentConfig(Config):
    """Configuration shared across training and inference."""

    NAME = "Component"
    BACKBONE = "mobilenetv1"
    NUM_CLASSES = len(CLASS_NAMES)
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50
    IMAGES_PER_GPU = 2
    DETECTION_MIN_CONFIDENCE = 0.8
    TRAIN_BN = None


class ComponentInferenceConfig(ComponentConfig):
    """Configuration tailored to inference on a single GPU/CPU."""

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def _get_skio():
    """Lazy load skimage.io to avoid import-time numpy compatibility issues."""
    global skio
    if skio is None:
        try:
            from skimage import io as skio_module
            skio = skio_module
        except (ImportError, AttributeError):
            # AttributeError can occur due to numpy binary compatibility issues
            skio = False  # Mark as attempted but failed
    return skio if skio is not False else None


@dataclass
class DetectionResult:
    """A light-weight view of the Mask R-CNN output."""

    rois: Sequence[Sequence[int]]
    class_ids: Sequence[int]
    scores: Sequence[float]
    masks: Any

    @classmethod
    def from_raw(cls, raw_result: Dict[str, Any]) -> "DetectionResult":
        return cls(
            rois=_to_sequence(raw_result.get("rois"), []),
            class_ids=_to_sequence(raw_result.get("class_ids"), []),
            scores=_to_sequence(raw_result.get("scores"), []),
            masks=raw_result.get("masks"),
        )


SequenceLike = Union[List[Any], Tuple[Any, ...]]


def _to_sequence(value: Any, default: SequenceLike) -> SequenceLike:
    if value is None:
        return list(default)
    if np is not None and hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return list(default)


def _infer_mask_depth(masks: Any) -> int:
    if masks is None:
        return 0
    if np is not None and isinstance(masks, np.ndarray):
        if masks.ndim < 3:
            return 0
        return int(masks.shape[2])
    try:
        first_row = masks[0]
        first_cell = first_row[0]
        return len(first_cell)
    except (IndexError, TypeError):
        return 0


def _slice_mask(masks: Any, index: int, height: int, width: int) -> List[List[bool]]:
    if np is not None and isinstance(masks, np.ndarray):
        return masks[:, :, index].astype(bool).tolist()
    slice_mask: List[List[bool]] = []
    for row in range(height):
        row_values: List[bool] = []
        for col in range(width):
            row_values.append(bool(masks[row][col][index]))
        slice_mask.append(row_values)
    return slice_mask


def _create_segmentation(height: int, width: int) -> List[List[int]]:
    return [[0 for _ in range(width)] for _ in range(height)]


def _finalise_segmentation(segmentation: List[List[int]]) -> Any:
    if np is not None:
        return np.array(segmentation, dtype=np.uint8)
    return segmentation


def _infer_image_shape(image: Any) -> Tuple[int, int, int]:
    if np is not None and hasattr(image, "shape"):
        shape = tuple(int(i) for i in image.shape)
        if len(shape) == 2:
            return shape + (1,)
        return shape  # type: ignore[return-value]

    if not isinstance(image, list):
        raise ValueError("Images passed to segment() must be list-like when numpy is unavailable")
    height = len(image)
    width = len(image[0]) if height else 0
    if height and isinstance(image[0][0], list):
        channels = len(image[0][0])
    else:
        channels = 1
    return height, width, channels


def _ensure_numpy_array(image: Any) -> Any:
    if np is not None and not isinstance(image, np.ndarray):
        return np.array(image)
    return image


def _validate_image_dimensions(image: Any) -> None:
    if np is not None and isinstance(image, np.ndarray):
        ndim = image.ndim
    else:
        ndim = _infer_list_ndim(image)
    if ndim not in (2, 3):
        raise ValueError("Images passed to segment() must be 2-D or 3-D arrays")


def _infer_list_ndim(image: Any) -> int:
    if not isinstance(image, list):
        return 1
    if not image:
        return 2
    if isinstance(image[0], list):
        if image[0] and isinstance(image[0][0], list):
            return 3
        return 2
    return 1


def build_segmentation_map(
    result: DetectionResult, image_shape: Tuple[int, int, int]
) -> Any:
    """Convert instance masks to a dense segmentation map."""

    height, width = image_shape[:2]
    segmentation = _create_segmentation(height, width)

    mask_depth = _infer_mask_depth(result.masks)
    if mask_depth == 0:
        return _finalise_segmentation(segmentation)

    class_ids = list(result.class_ids)
    for index, class_id in enumerate(class_ids):
        if index >= mask_depth:
            break
        if int(class_id) >= len(CLASS_NAMES):
            raise ValueError(f"Invalid class id {class_id} encountered during inference")
        mask_slice = _slice_mask(result.masks, index, height, width)
        for row in range(height):
            for col in range(width):
                if mask_slice[row][col]:
                    segmentation[row][col] = int(class_id)

    return _finalise_segmentation(segmentation)


class ComponentSegmenter:
    """High level helper class for performing inference."""

    def __init__(
        self,
        weights_path: Optional[str],
        *,
        model_dir: Optional[str] = None,
        detection_min_confidence: Optional[float] = None,
        model: Optional[Any] = None,
        load_weights: bool = True,
    ) -> None:
        self.config = ComponentInferenceConfig()
        if detection_min_confidence is not None:
            self.config.DETECTION_MIN_CONFIDENCE = detection_min_confidence

        if load_weights:
            if not weights_path:
                raise ValueError("A weight path must be supplied when load_weights=True")
        self.model = model if model is not None else self._build_model(model_dir)

        if load_weights:
            self.model.load_weights(weights_path, by_name=True)

    def _build_model(self, model_dir: Optional[str]) -> Any:
        from mrcnn import model as modellib  # Local import to avoid tensorflow dependency for tests

        return modellib.MaskRCNN(
            mode="inference",
            config=self.config,
            model_dir=model_dir or os.path.dirname(os.path.abspath(__file__)),
        )

    def segment(self, image: Any) -> Tuple[DetectionResult, Any]:
        _validate_image_dimensions(image)
        prepared_image = _ensure_numpy_array(image)
        raw_result = self.model.detect([prepared_image], verbose=0)[0]
        detection = DetectionResult.from_raw(raw_result)
        segmentation = build_segmentation_map(detection, _infer_image_shape(prepared_image))
        return detection, segmentation

    def segment_image_path(self, image_path: str) -> Tuple[DetectionResult, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        skio_module = _get_skio()
        if skio_module is None:
            raise RuntimeError("scikit-image is required to load images from disk")
        image = skio_module.imread(image_path)
        return self.segment(image)


def save_mask(mask: Any, output_path: str) -> None:
    """Persist the segmentation map to disk."""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    extension = os.path.splitext(output_path)[1].lower()
    if extension == ".npy":
        if np is not None:
            np.save(output_path, mask)
        else:
            with open(output_path, "wb") as file:
                pickle.dump(mask, file)
        return

    skio_module = _get_skio()
    if skio_module is None or np is None:
        raise RuntimeError("Saving masks as images requires numpy and scikit-image to be installed")

    skio_module.imsave(output_path, np.array(mask, dtype=np.uint8))


def parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone component segmentation inference")
    parser.add_argument("--weights", required=True, help="Path to the Mask R-CNN weights file (.h5)")
    parser.add_argument("--image", required=True, help="Path to the image to segment")
    parser.add_argument(
        "--output",
        help="Where to store the resulting mask. Defaults to '<image>_mask.png'.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Override the detection confidence threshold (default: 0.8)",
    )
    return parser.parse_args(args)


def main(cli_args: Optional[Iterable[str]] = None) -> None:
    args = parse_args(cli_args)

    default_output = os.path.splitext(args.image)[0] + "_mask.png"
    output_path = args.output or default_output

    segmenter = ComponentSegmenter(
        args.weights,
        detection_min_confidence=args.confidence,
    )
    _, segmentation = segmenter.segment_image_path(args.image)
    save_mask(segmentation, output_path)
    print(f"Mask written to {output_path}")


if __name__ == "__main__":  # pragma: no cover - exercised via CLI tests instead
    main()
