from pathlib import Path
import tempfile
import unittest

import numpy as np
import tifffile
import zarr

import scroll_to_ome


CASES = (
    ("uint16_to_uint8_full", 2, "uint16", 1,
     [[0, 256], [65280, 65535]], "uint8", [[0, 1], [255, 255]]),
    ("uint16_to_uint8_partial", 3, "uint16", 1,
     [[0, 256], [65280, 65535]], "uint8", [[0, 1], [255, 255]]),
    ("uint16_preserved_full", 2, "uint16", 0,
     [[0, 256], [65280, 65535]], "uint16", [[0, 256], [65280, 65535]]),
    ("uint16_preserved_partial", 3, "uint16", 0,
     [[0, 256], [65280, 65535]], "uint16", [[0, 256], [65280, 65535]]),
    ("uint8_preserved_full", 2, "uint8", 1,
     [[0, 1], [254, 255]], "uint8", [[0, 1], [254, 255]]),
    ("uint8_preserved_partial", 3, "uint8", 1,
     [[0, 1], [254, 255]], "uint8", [[0, 1], [254, 255]]),
)


class Tifs2ZarrTests(unittest.TestCase):
    def test_final_block_values(self):
        for name, planes, input_dtype, obytes, pattern, output_dtype, expected in CASES:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                tiffdir = root / "tiffs"
                tiffdir.mkdir()
                for plane in range(planes):
                    tifffile.imwrite(
                        tiffdir / f"{plane:03d}.tif",
                        np.array(pattern, dtype=input_dtype),
                        photometric="minisblack",
                        compression=None,
                    )

                zarrdir = root / "converted.zarr"
                result = scroll_to_ome.tifs2zarr(
                    tiffdir, zarrdir, chunk_size=2, obytes=obytes,
                    slices=None, maxgb=None,
                )
                self.assertIsNone(result)
                reopened = zarr.open(str(zarrdir), mode="r")
                self.assertEqual(reopened.shape, (planes, 2, 2))
                self.assertEqual(reopened.dtype, np.dtype(output_dtype))
                np.testing.assert_array_equal(reopened[:], [expected] * planes)


if __name__ == "__main__":
    unittest.main()
