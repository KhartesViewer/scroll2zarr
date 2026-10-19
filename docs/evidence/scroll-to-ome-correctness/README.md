# Reproducing the final-block and X/Y crop fixes

This note accompanies the pull request that removes a second division of the final converted block and applies X and Y crops independently. The files under `saved-output/` are excerpts of earlier saved runs. They are not new executions. Only isolated-run directory prefixes were replaced with `[isolated-run]`. Warnings and results are unchanged.

## Base and environment

- Base: KhartesViewer/scroll2zarr `fa373e3ce08c99483e906c4dcdcd0c95a89c6b81`. Its `scroll_to_ome.py` is 17,186 bytes with SHA256 `ddcada2c481b0dae27c794b310aa776d2756dab038f8b9d1eabf0f00ddf495fa`, the same bytes as the tested ScrollPrize/scroll2zarr `56c0bd9c83ea071ef41210497ba5faacebb01613`.
- Later recorded, compatible reproduction versions: CPython 3.12.12, numpy 2.5.3, zarr 2.18.7, tifffile 2026.9.20, numcodecs 0.15.1. The original correctness-run package versions were not recorded and remain unknown. These later versions are not a historical correctness-run environment record, a minimum or a lock file. Zarr 2 is required because the converter uses `zarr.NestedDirectoryStore`.
- Thread-count variables such as `OMP_NUM_THREADS` were set to 1.

## Command

From a checkout with the patch applied:

```sh
python -m unittest discover -s tests -p 'test_scroll_to_ome_*.py' -v
```

Both test modules write tiny synthetic TIFFs into a temporary directory, run `tifs2zarr`, and reopen the Zarr 2 output to check shape, dtype and every value. No scan data is needed.

## Expected values

Final block. A UInt16 plane `[[0,256],[65280,65535]]` converted to UInt8 should read `[[0,1],[255,255]]`. On the base, a full final block (two planes, chunk size two) and a partial final block (three planes) contain zeros where the last buffered block was divided twice. UInt16 and UInt8 preservation controls keep their values on both.

Crops. Each plane holds `40*z + 10*y + x` in four rows by three columns. Range strings are X,Y,Z and output axes are Z,Y,X.

| Range | Expected | Base |
| --- | --- | --- |
| `,1:3,1:4` (Y only) | shape `(3,2,3)`, first plane `[[50,51,52],[60,61,62]]` | starts `[[40,41,42],[50,51,52]]`, all 18 values wrong |
| `1:3,,1:4` (X only) | shape `(3,4,2)`, first plane `[[41,42],[51,52],[61,62],[71,72]]` | `ValueError` broadcasting `(4,3)` into `(4,2)` |

The crop tests pass `obytes=0` so they check slicing alone.

## Saved results

| Source | Result | File |
| --- | --- | --- |
| Base, both modules | full and partial final block FAIL, Y-only crop FAIL, X-only crop ERROR; `FAILED (failures=3, errors=1)` | [`saved-output/original.stderr.txt`](saved-output/original.stderr.txt) |
| Final-block fix, final-block module | six cases, `OK` | [`saved-output/final-only.stderr.txt`](saved-output/final-only.stderr.txt) |
| Crop fix, crop module | six cases, `OK` | [`saved-output/crop-only.stderr.txt`](saved-output/crop-only.stderr.txt) |
| Both fixes, both modules | 2 test methods, 12 cases as subtests, `OK` | [`saved-output/combined.stderr.txt`](saved-output/combined.stderr.txt) |

Each test method runs its six cases as subtests, so unittest reports one test per module. These are development tests. They make no claim about a real CT pipeline, a manual run or upstream CI.
