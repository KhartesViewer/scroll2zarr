# Staging-buffer memory evidence

This note accompanies the pull request that allocates the TIFF staging buffer in the output dtype. Everything here is a copy of an earlier saved run. Nothing was rerun to build it.

## What was measured

One synthetic workload, four fresh processes in ABBA order: reference, candidate, candidate, reference.

- Fixture. 32 UInt16 TIFF planes of 2048 x 2048, value `(257*z + 17*y + 31*x) % 65536`, uncompressed, 64 rows per strip. Written by [`historical-source/fixture.py`](historical-source/fixture.py).
- Measured function. `tifs2zarr` only. [`historical-source/producer.py`](historical-source/producer.py) parses the converter with `ast`, selects that one function and runs it with numpy, tifffile and zarr supplied. The rest of the module and the CLI were not exercised.
- Invocation. [`historical-source/arm.py`](historical-source/arm.py) calls `tifs2zarr(fixture, output, 32, obytes=1, slices=None, maxgb=None)`, which converts UInt16 to UInt8 with chunk size 32.
- Reference. [`historical-source/reference_fixed_scroll_to_ome.py`](historical-source/reference_fixed_scroll_to_ome.py) is the base converter with the final-block fix from the preceding pull request. It predates the crop fix, which does not touch the buffer.
- Candidate. [`historical-source/candidate_scroll_to_ome.py`](historical-source/candidate_scroll_to_ome.py) is the reference plus exactly the two lines of this pull request.
- Peak memory. Each arm reads `VmHWM` from `/proc/self/status` immediately after conversion ([`historical-source/common.py`](historical-source/common.py)). The supervisor ([`historical-source/driver.py`](historical-source/driver.py)) also records `ru_maxrss` from `wait4`. Both are Linux RSS high-water marks with shared accounting, so they are two reports of one quantity, not independent measurements.
- Conditions. Linux, one CPU, no swap, thread-count variables set to 1, CPython 3.12.12, numpy 2.5.3, zarr 2.18.7, tifffile 2026.9.20, numcodecs 0.15.1. The fixture was streamed through the page cache before every arm. Parameters are in [`historical-source/protocol.json`](historical-source/protocol.json).

## Results

| Arm | Role | VmHWM bytes | wait4 bytes |
| --- | --- | ---: | ---: |
| a1 | reference | 379,719,680 | 379,719,680 |
| a2 | candidate | 245,530,624 | 245,530,624 |
| a3 | candidate | 245,506,048 | 245,506,048 |
| a4 | reference | 379,797,504 | 379,797,504 |

The comparison uses the lowest reference against the highest candidate. That is 379,719,680 − 245,530,624 = 134,189,056 bytes, 127.97 MiB or 35.34% of the reference peak. The expected buffer change is 32 × 2048 × 2048 × 1 byte = 134,217,728 bytes (128 MiB), within 28,672 bytes of the observed difference. Machine-readable rows are in [`peak-rss.csv`](peak-rss.csv). Per-arm records are [`saved-results/a1.json`](saved-results/a1.json) through [`saved-results/a4.json`](saved-results/a4.json) and [`saved-results/a1-wait4.json`](saved-results/a1-wait4.json) through [`saved-results/a4-wait4.json`](saved-results/a4-wait4.json). The `a*.json` records list the SHA256 of every source file above.

## Output equality

[`historical-source/checker.py`](historical-source/checker.py) recomputed the expected UInt8 value `((257*z + 17*y + 31*x) % 65536) // 256` for all 134,217,728 coordinates of every arm and required an exact match. It then hashed every persisted output file. All four arms wrote 4,097 files with identical hashes. [`saved-results/output-equality.json`](saved-results/output-equality.json) restates that from the saved checker receipt. The full receipt is 1.67 MB and is not included. Its SHA256 is listed there. [`saved-results/fixture.json`](saved-results/fixture.json) seals the 32 input TIFFs, and [`saved-results/completion.json`](saved-results/completion.json) is the run's completion record.

## Compatibility with the other fixes

[`saved-output/memory-compatibility.stderr.txt`](saved-output/memory-compatibility.stderr.txt) is a separate later run of the final composition of all three changes. Three test methods passed: the two portable correctness modules and a local banded check of explicit `maxgb`, multiple Y bands and UInt16 preservation. That banded check depends on a local reference module and is not included here. This run was not a memory measurement.

## Limits

Local paths in the saved records are replaced with `[venv]`, `[cpython-3.12.12]`, `[run-dir]`, `[historical-source]` and `[private-records]`. The historical source files are exact copies, but they ran inside a resource-limited supervisor that is not included. They are a record of what ran, not a portable benchmark. The result covers this one workload with two repetitions per arm. It does not establish a general percentage, speed, `maxgb` compliance or behaviour on real CT volumes.
