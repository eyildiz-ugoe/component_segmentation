# Component Segmentation

This repository contains a standalone Python implementation for running
component segmentation using a Mask R-CNN model.  The original project shipped
with ROS integration and extensive bookkeeping utilities.  Those dependencies
have been removed in favour of a lightweight script that consumes an image and
produces a segmentation mask.

## Getting started

1. **Install dependencies**

   Create and activate a virtual environment, then install the Python
   dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   The repository vendors the Mask R-CNN implementation so no additional
   packages are required beyond the list above.

2. **Download the pre-trained weights**

   Download the `model.h5` file from
   [Google Drive](https://drive.google.com/file/d/1wj0FzDZ_niXflA38LqRaNpd0OQkt1HAe/view?usp=sharing)
   and place it in a convenient location on disk.

3. **Run inference**

   ```bash
   python component.py --weights /path/to/model.h5 --image /path/to/input.jpg \
       --output /path/to/output_mask.png
   ```

   If the `--output` argument is omitted the script will default to writing a
   PNG file next to the input image whose name ends with `_mask.png`.  Masks can
   alternatively be saved as NumPy arrays by using a path that ends with
   `.npy`.

## Python API

The `ComponentSegmenter` class exposes the same functionality programmatically:

```python
from component import ComponentSegmenter

segmenter = ComponentSegmenter("/path/to/model.h5")
_, mask = segmenter.segment_image_path("/path/to/input.jpg")
```

The returned `mask` is a two-dimensional `numpy.ndarray` whose values correspond
to indices in `component.CLASS_NAMES`.

## Testing

Run the unit tests with `pytest`:

```bash
pytest tests
```

The tests use a light-weight stub model and do not require the actual Mask
R-CNN weights.
