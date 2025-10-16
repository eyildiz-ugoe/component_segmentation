import importlib
import pickle

import pytest


@pytest.fixture(scope="module")
def component_module():
    return importlib.import_module("component")


def _make_mask(height, width, instance_count):
    return [[[False for _ in range(instance_count)] for _ in range(width)] for _ in range(height)]


def test_build_segmentation_map_assigns_class_ids(component_module):
    DetectionResult = component_module.DetectionResult
    build_segmentation_map = component_module.build_segmentation_map

    height, width = 4, 4
    masks = _make_mask(height, width, 2)
    for row in range(0, 2):
        for col in range(0, 2):
            masks[row][col][0] = True
    for row in range(2, 4):
        for col in range(2, 4):
            masks[row][col][1] = True

    result = DetectionResult(
        rois=[[0, 0, 1, 1], [2, 2, 3, 3]],
        class_ids=[1, 3],
        scores=[0.99, 0.75],
        masks=masks,
    )

    segmentation = build_segmentation_map(result, (height, width, 3))
    top_left = segmentation[0][0]
    bottom_right = segmentation[3][3]
    assert int(top_left) == 1
    assert int(bottom_right) == 3
    unique_values = {int(value) for row in segmentation for value in row}
    assert unique_values == {0, 1, 3}


def test_build_segmentation_map_rejects_unknown_class(component_module):
    DetectionResult = component_module.DetectionResult
    build_segmentation_map = component_module.build_segmentation_map
    CLASS_NAMES = component_module.CLASS_NAMES

    height, width = 2, 2
    masks = _make_mask(height, width, 1)
    for row in range(height):
        for col in range(width):
            masks[row][col][0] = True
    result = DetectionResult(
        rois=[[0, 0, 1, 1]],
        class_ids=[len(CLASS_NAMES) + 1],
        scores=[0.5],
        masks=masks,
    )

    with pytest.raises(ValueError):
        build_segmentation_map(result, (height, width, 3))


def test_component_segmenter_segment_uses_model(component_module):
    ComponentSegmenter = component_module.ComponentSegmenter
    DetectionResult = component_module.DetectionResult

    image = [[[0, 0, 0] for _ in range(4)] for _ in range(4)]
    masks = _make_mask(4, 4, 1)
    for row in range(4):
        for col in range(4):
            masks[row][col][0] = True
    raw_result = {
        "rois": [[0, 0, 1, 1]],
        "class_ids": [2],
        "scores": [0.9],
        "masks": masks,
    }

    class DummyModel:
        def __init__(self, result):
            self.result = result
            self.detect_calls = 0

        def load_weights(self, path, by_name=True):  # pragma: no cover - not used directly
            self.loaded_weights = path

        def detect(self, images, verbose=0):
            self.detect_calls += 1
            return [self.result]

    dummy = DummyModel(raw_result)
    segmenter = ComponentSegmenter(
        None,
        model=dummy,
        load_weights=False,
    )

    detection, segmentation = segmenter.segment(image)
    assert dummy.detect_calls == 1
    assert list(detection.class_ids) == [2]
    assert len(segmentation) == 4
    assert all(int(value) == 2 for row in segmentation for value in row)


def test_segment_image_path_reads_image(component_module, monkeypatch, tmp_path):
    ComponentSegmenter = component_module.ComponentSegmenter

    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"placeholder")

    captured = {"read": False}

    class FakeSkio:
        @staticmethod
        def imread(path):
            captured["read"] = True
            assert path == str(image_path)
            return [[[255, 255, 255] for _ in range(8)] for _ in range(8)]

    raw_result = {
        "rois": [[0, 0, 7, 7]],
        "class_ids": [1],
        "scores": [0.95],
        "masks": _make_mask(8, 8, 1),
    }
    for row in raw_result["masks"]:
        for col in row:
            col[0] = True

    class DummyModel:
        def __init__(self, result):
            self.result = result

        def detect(self, images, verbose=0):
            return [self.result]

    dummy = DummyModel(raw_result)

    monkeypatch.setattr("component.skio", FakeSkio, raising=False)

    segmenter = ComponentSegmenter(
        None,
        model=dummy,
        load_weights=False,
    )

    _, segmentation = segmenter.segment_image_path(str(image_path))
    assert captured["read"] is True
    assert len(segmentation) == 8
    assert all(int(value) == 1 for row in segmentation for value in row)


def test_save_mask_numpy(component_module, tmp_path):
    save_mask = component_module.save_mask

    mask = [[i * 4 + j for j in range(4)] for i in range(4)]
    out_path = tmp_path / "mask.npy"

    save_mask(mask, str(out_path))

    with open(out_path, "rb") as file:
        loaded = pickle.load(file)

    assert loaded == mask


def test_segment_requires_weights_when_loading(component_module):
    ComponentSegmenter = component_module.ComponentSegmenter

    with pytest.raises(ValueError):
        ComponentSegmenter("", load_weights=True)


def test_segment_rejects_invalid_dimensions(component_module):
    ComponentSegmenter = component_module.ComponentSegmenter

    class DummyModel:
        def detect(self, images, verbose=0):
            return [{"rois": [], "class_ids": [], "scores": [], "masks": _make_mask(4, 4, 0)}]

    segmenter = ComponentSegmenter(None, model=DummyModel(), load_weights=False)

    with pytest.raises(ValueError):
        segmenter.segment([0, 0, 0, 0])
