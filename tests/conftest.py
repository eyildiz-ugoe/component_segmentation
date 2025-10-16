import pathlib
import sys
import types


def _array(data, dtype=None):
    if isinstance(data, (list, tuple)):
        return [_array(item, dtype=dtype) for item in data]
    return data


fake_numpy = types.ModuleType("numpy")
fake_numpy.IS_FAKE = True
fake_numpy.array = _array
fake_numpy.ndarray = list

sys.modules.setdefault("numpy", fake_numpy)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
