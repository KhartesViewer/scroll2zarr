import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tifffile
import zarr


SOURCE = Path(__file__).resolve().parents[1] / "scroll_to_ome.py"
spec = importlib.util.spec_from_file_location("scroll_to_ome_under_test", SOURCE)
scroll_to_ome = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scroll_to_ome)

INPUT = [
    [[0, 1, 2], [10, 11, 12], [20, 21, 22], [30, 31, 32]],
    [[40, 41, 42], [50, 51, 52], [60, 61, 62], [70, 71, 72]],
    [[80, 81, 82], [90, 91, 92], [100, 101, 102], [110, 111, 112]],
    [[120, 121, 122], [130, 131, 132], [140, 141, 142], [150, 151, 152]],
]
Y_ONLY = [
    [[50, 51, 52], [60, 61, 62]],
    [[90, 91, 92], [100, 101, 102]],
    [[130, 131, 132], [140, 141, 142]],
]
X_ONLY = [
    [[41, 42], [51, 52], [61, 62], [71, 72]],
    [[81, 82], [91, 92], [101, 102], [111, 112]],
    [[121, 122], [131, 132], [141, 142], [151, 152]],
]
BOTH = [
    [[51, 52], [61, 62]],
    [[91, 92], [101, 102]],
    [[131, 132], [141, 142]],
]
CASES = (
    ("uint16_y_only_partial", "uint16", ",1:3,1:4", (3, 2, 3), Y_ONLY, [1, 2, 3]),
    ("uint16_x_only_partial", "uint16", "1:3,,1:4", (3, 4, 2), X_ONLY, [1, 2, 3]),
    ("uint16_no_crop", "uint16", None, (4, 4, 3), INPUT, [0, 1, 2, 3]),
    ("uint16_both_partial", "uint16", "1:3,1:3,1:4", (3, 2, 2), BOTH, [1, 2, 3]),
    ("uint8_no_crop", "uint8", None, (4, 4, 3), INPUT, [0, 1, 2, 3]),
    ("uint8_both_partial", "uint8", "1:3,1:3,1:4", (3, 2, 2), BOTH, [1, 2, 3]),
)


class Tifs2ZarrTests(unittest.TestCase):
    def test_spatial_crop_values(self):
        for name, dtype, ranges, shape, expected, z_ids in CASES:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                tiffdir = root / "tiffs"
                tiffdir.mkdir()
                for z, plane in enumerate(INPUT):
                    tifffile.imwrite(
                        tiffdir / f"{z:03d}.tif", np.array(plane, dtype=dtype),
                        photometric="minisblack", compression=None,
                    )
                slices = None if ranges is None else scroll_to_ome.parseSlices(ranges)
                zarrdir = root / "converted.zarr"
                result = scroll_to_ome.tifs2zarr(
                    tiffdir, zarrdir, chunk_size=2, obytes=0,
                    slices=slices, maxgb=None,
                )
                self.assertIsNone(result)
                reopened = zarr.open(str(zarrdir), mode="r")
                self.assertEqual(reopened.shape, shape)
                self.assertEqual(reopened.dtype, np.dtype(dtype))
                np.testing.assert_array_equal(reopened[:], expected)
                self.assertEqual((reopened[:, 0, 0] // 40).tolist(), z_ids)


if __name__ == "__main__":
    unittest.main()
