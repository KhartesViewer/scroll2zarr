import sys
import re
from pathlib import Path
import json
import shutil
import argparse
import copy
import numpy as np
import tifffile
import zarr
import fsspec
import warnings
from numcodecs.registry import codec_registry
import skimage.transform
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from scipy import ndimage


def init_o2z_globals():
    # Suppress warning about NestedDirectoryStore being removed
    # in zarr version 3.
    warnings.filterwarnings("ignore", message="The NestedDirectoryStore.*")
    # Per https://github.com/zarr-developers/zarr-python/blob/v2.18.3/zarr/storage.py :
    """
    NestedDirectoryStore will be removed in Zarr-Python 3.0 where controlling
    the chunk key encoding will be supported as part of the array metadata. See
    `GH1274 <https://github.com/zarr-developers/zarr-python/issues/1274>`_
    for more information.
    """

    # Controls the number of open https connections;
    # if this is not set, the Vesuvius Challenge data server
    # may complain of too many requests
    # https://filesystem-spec.readthedocs.io/en/latest/async.html
    fsspec.config.conf["nofiles_gather_batch_size"] = 10


class DecompressedLRUCache(zarr.storage.LRUStoreCache):
    def __init__(self, store, max_size):
        super().__init__(store, max_size)
        self.compressor = None

    # By default, the LRU cache holds chunks that it copies
    # directly from the original data store.  This means that
    # if the data store contains compressed chunks, the cache
    # will hold compressed chunks.
    # Each such chunk has to be decompressed every time it is
    # accessed, which is a waste of CPU.
    # This causes noticeable slowing.
    # The routine below modifies the internals of the array
    # that uses the LRU cache as the data store, so that compressed
    # chunks are decompressed when they go into the cache, and
    # so that they are not decompressed an additional time when
    # they are accessed.
    def transferCompressor(self, array):
        self.compressor = array._compressor
        array._compressor = None

    # This function gets a chunk from the underlying data
    # store.  The access may cause an exception to be thrown.
    # This function does not try to catch exceptions, because
    # the caller of this function will handle them.
    # If decompression has been transfered to the LRU cache
    # (see the transferCompressor function), do the decompression
    # here.
    def getValue(self, key):
        value = self._store[key]
        if len(value) > 0 and self.compressor is not None:
            for i in range(3):
                try:
                    dc = self.compressor.decode(value)
                    break
                except Exception as e:
                    print("decompression failure try %d: %s" % (i + 1, e))
            return dc
        return value

    # This is identical to __getitem__ in LRUStoreCache,
    # except that the access to self._store is replaced
    # by a call to self.getValue
    def __getitem__(self, key):
        try:
            # first try to obtain the value from the cache
            with self._mutex:
                value = self._values_cache[key]
                # cache hit if no KeyError is raised
                self.hits += 1
                # treat the end as most recently used
                self._values_cache.move_to_end(key)

        except KeyError:
            # cache miss, retrieve value from the store
            # value = self._store[key]
            value = self.getValue(key)
            with self._mutex:
                self.misses += 1
                # need to check if key is not in the cache,
                # as it may have been cached
                # while we were retrieving the value from the store
                if key not in self._values_cache:
                    self._cache_value(key, value)

        return value


# create ome dir, .zattrs, .zgroup
# (don't need to know output array dimensions, just number of levels,
# possibly unit/dimension info)
# create_ome_dir(zarrdir, nlevels)
# quit if dir already exists


def parseShift(istr):
    sstrs = istr.split(",")
    if len(sstrs) != 3:
        print(
            "Could not parse shift argument '%s'; expected 3 comma-separated numbers"
            % istr
        )
        return None
    shift = []
    for sstr in sstrs:
        i = int(sstr)
        shift.append(i)
    return shift


def parseSlices(istr):
    sstrs = istr.split(",")
    if len(sstrs) != 3:
        print(
            "Could not parse range argument '%s'; expected 3 comma-separated ranges"
            % istr
        )
        return (None, None, None)
    slices = []
    for sstr in sstrs:
        if sstr == "":
            slices.append(None)
            continue
        parts = sstr.split(":")
        if len(parts) == 1:
            slices.append(slice(int(parts[0])))
        else:
            iparts = [None if p == "" else int(p) for p in parts]
            if len(iparts) == 2:
                iparts.append(None)
            slices.append(slice(iparts[0], iparts[1], iparts[2]))
    return slices


def slice_step_is_1(s):
    if s is None:
        return True
    if s.step is None:
        return True
    if s.step == 1:
        return True
    return False


def slice_start(s, default=0):
    if s is None or s.start is None:
        return default
    return s.start


def slice_stop(s, default=0):
    if s is None or s.stop is None:
        return default
    return s.stop


def divp1(s, c):
    n = s // c
    if s % c > 0:
        n += 1
    return n


# return None if succeeds, err string if fails
def create_ome_dir(zarrdir):
    # complain if directory already exists
    if zarrdir.exists():
        err = "Directory %s already exists" % zarrdir
        print(err)
        return err

    try:
        zarrdir.mkdir()
    except Exception as e:
        err = "Error while creating %s: %s" % (zarrdir, e)
        print(err)
        return err


def create_ome_headers(zarrdir, nlevels, preserve_z=False):
    zattrs_dict = {
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [],
                "name": "/",
                "version": "0.4",
            }
        ]
    }

    dataset_dict = {
        "coordinateTransformations": [{"scale": [], "type": "scale"}],
        "path": "",
    }

    zgroup_dict = {"zarr_format": 2}

    datasets = []
    for l in range(nlevels):
        ds = copy.deepcopy(dataset_dict)
        ds["path"] = "%d" % l

        if preserve_z:
            xy_scale = 2.0**l
            z_scale = 1.0
            ds["coordinateTransformations"][0]["scale"] = [z_scale, xy_scale, xy_scale]
        else:
            scale = 2.0**l
            ds["coordinateTransformations"][0]["scale"] = [scale] * 3
            # print(json.dumps(ds, indent=4))

        datasets.append(ds)
    zad = copy.deepcopy(zattrs_dict)
    zad["multiscales"][0]["datasets"] = datasets
    json.dump(zgroup_dict, (zarrdir / ".zgroup").open("w"), indent=4)
    json.dump(zad, (zarrdir / ".zattrs").open("w"), indent=4)


def path_is_file(path):
    fs, _, _ = fsspec.get_fs_token_paths(path)
    proto = fs.protocol
    print("fsspec protocol(s):", proto)
    return "file" in proto


# compression is None or a string.  None means use the
# compression of the input zarr, a string of "none" or "" means
# no compression; otherwise the string gives the name of the compression
# scheme.
# If chunk_size is None, the input zarr's chunking is used, otherwise
# chunks of (chunk_size, chunk_size, chunk_size) are used.
# If dtype is None, the input zarr's dtype is used.  Otherwise the
# given dtype is used.  Note that the only supported conversion
# is from uint16 to uint8.
def zarr2zarr(
    izarrdir,
    ozarrdir,
    shift=(0, 0, 0),
    slices=(None, None, None),
    chunk_size=None,
    compression=None,
    dtype=None,
    channel=0,
):
    # user gives shift and slices in x,y,z order, but internally
    # we use z,y,x order
    shift = (shift[2], shift[1], shift[0])
    slices = (slices[2], slices[1], slices[0])
    # slice_start >= 0 for all slices
    if not all([(slice_start(s) >= 0) for s in slices]):
        err = "All window starting coordinates must be >= 0"
        print(err)
        return err
    if not all([slice_step_is_1(s) for s in slices]):
        err = "All window steps must be 1"
        print(err)
        return err
    try:
        izarr = zarr.open(izarrdir, mode="r")
    except Exception as e:
        err = "Could not open %s; error is %s" % (izarrdir, e)
        print(err)
        return err
    is_ome = False
    if isinstance(izarr, zarr.hierarchy.Group):
        print(
            "It appears that",
            izarrdir,
            "\nis an OME-Zarr store rather than a simple zarr store.\nI will attempt to open the highest-resolution zarr store in this hierarchy\n",
        )
        if "0" in izarr:
            izarr = izarr["0"]
        else:
            err = "%s does not appear to be a zarr or OME-Zarr data store"
            print(err)
            return err
        is_ome = True

    # Check if the input zarr has 4 dimensions (with channel)
    has_channel = len(izarr.shape) == 4
    if has_channel:
        print(f"Detected 4D input zarr with shape {izarr.shape}")
        print(f"Extracting channel {channel}")
        if channel >= izarr.shape[0]:
            err = f"Channel index {channel} is out of bounds for shape {izarr.shape}"
            print(err)
            return err
        # Adjust chunks to remove channel dimension
        chunk_sizes = izarr.chunks[1:]  # Remove the first (channel) dimension
    else:
        chunk_sizes = izarr.chunks

    if chunk_size is not None:
        chunk_sizes = (chunk_size, chunk_size, chunk_size)
    divisor = 1
    if dtype is None:
        dtype = izarr.dtype
    elif dtype != izarr.dtype:
        if dtype == np.uint8 and izarr.dtype == np.uint16:
            divisor = 256
        else:
            err = "Can't convert %s to %s" % (str(izarr.dtype), str(dtype))
            print(err)
            return err

    compressor = izarr.compressor
    if compression is not None:
        if compression == "none" or compression == "":
            compressor = None
        else:
            codec_cls = codec_registry[compression]
            compressor = codec_cls()
            # if compression_opts is dict:
            # compressor = codec_cls(**compression_opts)

    # If izarrdir is not a file (ie if it is a url for streaming),
    # then setting up the cache is very slow (the zarr.open line
    # below takes a very long time).  So in the case of a url better
    # to skip caching
    if path_is_file(izarrdir):
        # TODO: allow larger max_size
        store = izarr.store
        print("Using cache for input data")
        cstore = DecompressedLRUCache(store, max_size=2**32)
        if is_ome:
            root = zarr.group(store=cstore)
            czarr = root["0"]
        else:
            czarr = zarr.open(cstore, mode="r")
        cstore.transferCompressor(czarr)
    else:
        czarr = izarr

    # Adjust shape if input has channel dimension
    if has_channel:
        ishape = izarr.shape[1:]  # Remove channel dimension
    else:
        ishape = izarr.shape

    # i0 and i1 are the min and max of the input zarr
    # before windowing and shifting
    i0 = [0, 0, 0]
    i1 = ishape
    # iw0 and iw1 are the min and max of the windowed
    # part of the input zarr before shifting
    iw0 = [slice_start(slices[i], i0[i]) for i in range(3)]
    iw1 = [slice_stop(slices[i], i1[i]) for i in range(3)]
    # TODO verify that iw1 is > iw0 ?

    # iws0 and iws1 are the min and max of the
    # windowed, shifted input zarr.
    iws0 = [iw0[i] + shift[i] for i in range(3)]
    iws1 = [iw1[i] + shift[i] for i in range(3)]
    # os0 and os1 are the min and max of the output
    # zarr in shifted coordinates
    os0 = [max(0, iws0[i]) for i in range(3)]
    os1 = [max(0, iws1[i]) for i in range(3)]
    # TODO verify that os0 < os1?
    if not all([os0[i] < os1[i] for i in range(3)]):
        err = "Computed conflicting output grid min %s and max %s" % (
            str(os0),
            str(os1),
        )
        print(err)
        return err

    oshape = os1

    # cs0 and cs1 are the min and max of the part of the input grid
    # that should be copied to the output grid,
    # in shifted (output-grid) coordinates
    cs0 = [max(os0[i], iws0[i]) for i in range(3)]
    cs1 = [min(os1[i], iws1[i]) for i in range(3)]

    store = zarr.NestedDirectoryStore(ozarrdir)
    ozarr = zarr.open(
        store=store,
        shape=oshape,
        chunks=chunk_sizes,
        dtype=dtype,
        write_empty_chunks=False,
        fill_value=0,
        compressor=compressor,
        mode="w",
    )

    chunk0s = [cs0[i] // chunk_sizes[i] for i in range(3)]
    # TODO: should chunk1s be incremented by 1?
    # chunk1s = [cs1[i] // chunk_sizes[i] for i in range(3)]
    chunk1s = [(cs1[i] - 1) // chunk_sizes[i] + 1 for i in range(3)]
    print("chunk_sizes", chunk_sizes)
    # print("chunk0s", chunk0s)
    # print("nchunks", nchunks)
    ci = [0, 0, 0]
    for ci[0] in range(chunk0s[0], chunk1s[0]):
        print("doing", ci[0], "to", chunk1s[0])
        for ci[1] in range(chunk0s[1], chunk1s[1]):
            for ci[2] in range(chunk0s[2], chunk1s[2]):
                # print(ci)
                o0 = [max(cs0[i], chunk_sizes[i] * ci[i]) for i in range(3)]
                o1 = [min(cs1[i], chunk_sizes[i] * (1 + ci[i])) for i in range(3)]
                i0 = [o0[i] - shift[i] for i in range(3)]
                i1 = [o1[i] - shift[i] for i in range(3)]
                if has_channel:
                    # Select the specified channel from 4D array
                    ozarr[o0[0] : o1[0], o0[1] : o1[1], o0[2] : o1[2]] = (
                        czarr[channel, i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2]]
                        // divisor
                    )
                else:
                    ozarr[o0[0] : o1[0], o0[1] : o1[1], o0[2] : o1[2]] = (
                        czarr[i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2]] // divisor
                    )


def process_chunk(args):
    idata, odata, z, y, x, cz, cy, cx, algorithm = args
    ibuf = idata[
        2 * z * cz : (2 * z * cz + 2 * cz),
        2 * y * cy : (2 * y * cy + 2 * cy),
        2 * x * cx : (2 * x * cx + 2 * cx),
    ]
    if np.max(ibuf) == 0:
        return  # Skip if the block is empty to save computation

    # pad ibuf to even in all directions
    ibs = ibuf.shape
    pad = (ibs[0] % 2, ibs[1] % 2, ibs[2] % 2)
    if any(pad):
        ibuf = np.pad(ibuf, ((0, pad[0]), (0, pad[1]), (0, pad[2])), mode="symmetric")

    # algorithms:
    if algorithm == "nearest":
        obuf = ibuf[::2, ::2, ::2]
    elif algorithm == "gaussian":
        obuf = np.round(skimage.transform.rescale(ibuf, 0.5, preserve_range=True))
    elif algorithm == "mean":
        obuf = np.round(skimage.transform.downscale_local_mean(ibuf, (2, 2, 2)))
    elif algorithm == "max":
        s = ibuf.shape
        ibuf = ibuf.reshape((s[0] // 2, 2, s[1] // 2, 2, s[2] // 2, 2))
        obuf = np.max(ibuf, axis=(1, 3, 5))
    else:
        raise ValueError(f"algorithm {algorithm} not valid")

    odata[
        z * cz : (z * cz + cz), y * cy : (y * cy + cy), x * cx : (x * cx + cx)
    ] = np.round(obuf)


def process_chunk_preserve_z(args):
    (
        idata,
        odata,
        z,
        y,
        x,
        cz,
        cy,
        cx,
        algorithm,
        xy_factor,
        apply_smoothing,
        binary_threshold,
    ) = args

    iz_start = z * cz
    iz_end = min((z + 1) * cz, idata.shape[0])
    iy_start = y * cy * xy_factor
    iy_end = min((y + 1) * cy * xy_factor, idata.shape[1])
    ix_start = x * cx * xy_factor
    ix_end = min((x + 1) * cx * xy_factor, idata.shape[2])

    ibuf = idata[iz_start:iz_end, iy_start:iy_end, ix_start:ix_end]

    if np.max(ibuf) == 0:
        return

    downsample_factor = int(xy_factor)

    if downsample_factor == 1:
        obuf = ibuf
    else:
        zlen, H, W = ibuf.shape
        pad_h = (-H) % downsample_factor
        pad_w = (-W) % downsample_factor
        if pad_h or pad_w:
            ibuf = np.pad(ibuf, ((0, 0), (0, pad_h), (0, pad_w)), mode="symmetric")
            H += pad_h
            W += pad_w

        Hn = H // downsample_factor
        Wn = W // downsample_factor

        if apply_smoothing or algorithm == "gaussian":
            sigma = 0.5 * downsample_factor
            tmp = ndimage.gaussian_filter(
                ibuf.astype(np.float32, copy=False),
                sigma=(0.0, sigma, sigma),
                mode="reflect",
            )
            obuf = tmp[:, ::downsample_factor, ::downsample_factor]

            if apply_smoothing and binary_threshold is not None:
                if 0.0 <= float(binary_threshold) <= 1.0 and np.issubdtype(
                    idata.dtype, np.integer
                ):
                    thr_val = float(binary_threshold) * np.iinfo(idata.dtype).max
                else:
                    thr_val = float(binary_threshold)

                mask = obuf > thr_val
                out_hi = (
                    np.iinfo(odata.dtype).max
                    if np.issubdtype(odata.dtype, np.integer)
                    else 1.0
                )
                obuf = np.where(mask, out_hi, 0).astype(odata.dtype, copy=False)
            else:
                obuf = obuf.astype(odata.dtype, copy=False)
        elif algorithm == "nearest":
            obuf = ibuf[:, ::downsample_factor, ::downsample_factor]
        elif algorithm == "mean":
            obuf = skimage.transform.downscale_local_mean(
                ibuf, (1, downsample_factor, downsample_factor)
            )
        elif algorithm == "max":
            obuf = ibuf.reshape(zlen, Hn, downsample_factor, Wn, downsample_factor).max(
                axis=(2, 4)
            )
        else:
            raise ValueError(f"algorithm {algorithm} not valid")

    oz_start = z * cz
    oy_start = y * cy
    ox_start = x * cx

    oz_end = min(oz_start + obuf.shape[0], odata.shape[0])
    oy_end = min(oy_start + obuf.shape[1], odata.shape[1])
    ox_end = min(ox_start + obuf.shape[2], odata.shape[2])

    oview = obuf[: oz_end - oz_start, : oy_end - oy_start, : ox_end - ox_start]

    if np.issubdtype(odata.dtype, np.integer) and oview.dtype.kind == "f":
        oview = np.rint(oview).astype(odata.dtype, copy=False)
    else:
        oview = oview.astype(odata.dtype, copy=False)

    odata[oz_start:oz_end, oy_start:oy_end, ox_start:ox_end] = oview


def resize(zarrdir, old_level, num_threads, algorithm="mean"):
    idir = zarrdir / ("%d" % old_level)
    if not idir.exists():
        err = f"input directory {idir} does not exist"
        print(err)
        return err

    odir = zarrdir / ("%d" % (old_level + 1))
    idata = zarr.open(idir, mode="r")
    print(
        "Creating level",
        old_level + 1,
        "  input array shape",
        idata.shape,
        " algorithm",
        algorithm,
    )

    cz, cy, cx = idata.chunks
    sz, sy, sx = idata.shape
    store = zarr.NestedDirectoryStore(odir)
    odata = zarr.open(
        store=store,
        shape=(divp1(sz, 2), divp1(sy, 2), divp1(sx, 2)),
        chunks=idata.chunks,
        dtype=idata.dtype,
        write_empty_chunks=False,
        fill_value=0,
        compressor=idata.compressor,
        mode="w",
    )

    # Prepare tasks
    tasks = [
        (idata, odata, z, y, x, cz, cy, cx, algorithm)
        for z in range(divp1(sz, 2 * cz))
        for y in range(divp1(sy, 2 * cy))
        for x in range(divp1(sx, 2 * cx))
    ]

    # Use ThreadPoolExecutor to process blocks in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(process_chunk, tasks), total=len(tasks)))

    print("Processing complete")


def resize_preserve_z(
    zarrdir,
    old_level,
    num_threads,
    algorithm="mean",
    enable_level_smoothing=False,
    smooth_start_level=4,
    binary_threshold=None,
):
    idir = zarrdir / ("%d" % old_level)
    if not idir.exists():
        err = f"input directory {idir} does not exist"
        print(err)
        return err

    odir = zarrdir / ("%d" % (old_level + 1))
    idata = zarr.open(idir, mode="r")

    xy_factor = 2
    apply_smoothing = enable_level_smoothing and (old_level + 1) >= smooth_start_level

    print(
        "Creating level",
        old_level + 1,
        "  input array shape",
        idata.shape,
        " algorithm",
        algorithm,
        " Z preserved, XY factor",
        xy_factor,
        " smoothing",
        apply_smoothing,
    )

    cz, cy, cx = idata.chunks
    sz, sy, sx = idata.shape
    store = zarr.NestedDirectoryStore(odir)
    odata = zarr.open(
        store=store,
        shape=(sz, divp1(sy, xy_factor), divp1(sx, xy_factor)),
        chunks=idata.chunks,
        dtype=idata.dtype,
        write_empty_chunks=False,
        fill_value=0,
        compressor=idata.compressor,
        mode="w",
    )

    tasks = [
        (
            idata,
            odata,
            z,
            y,
            x,
            cz,
            cy,
            cx,
            algorithm,
            xy_factor,
            apply_smoothing,
            binary_threshold,
        )
        for z in range(divp1(sz, cz))
        for y in range(divp1(divp1(sy, xy_factor), cy))
        for x in range(divp1(divp1(sx, xy_factor), cx))
    ]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(process_chunk_preserve_z, tasks), total=len(tasks)))

    print("Processing complete")


def zarr_to_ome_main():
    init_o2z_globals()
    # parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create OME/Zarr data store from an existing zarr data store",
        # allow_abbrev=False,
        # prefix_chars='--',
    )
    parser.add_argument("input_zarr_dir", help="Name of zarr store directory")
    parser.add_argument(
        "output_zarr_ome_dir",
        help="Name of directory that will contain OME/zarr datastore",
    )
    parser.add_argument(
        "--algorithm",
        choices=["mean", "gaussian", "nearest", "max"],
        default="mean",
        help="Algorithm used to sub-sample the data.  Use 'mean' if the input data is continuous, 'nearest' or 'max' if the input data represents an indicator",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        help="Size of chunk; if not given, will be same as input zarr",
    )
    parser.add_argument(
        "--shift",
        help="Shift input data set, relative to output data set.  Example (in xyz order): 2000,3000,5500.  If any of the shifts is negative, use '=' instead of space with flag, for example --shift=-1000,-2000,-3000 instead of --shift -1000,-2000,-3000",
    )
    parser.add_argument(
        "--window",
        help="Output only a subset of the data.  Example (in xyz order, and based on the output-data grid): 2500:3000,1500:4000,500:600",
    )
    parser.add_argument(
        "--bytes",
        type=int,
        default=0,
        help="number of bytes per pixel in output; if not given, will be same as input zarr",
    )
    parser.add_argument(
        "--compression",
        help="Compression algorithm ('blosc', 'none', ...); if not given, will be same as input zarr; if set to 'none', no compression will be used",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory, if it already exists",
    )
    parser.add_argument(
        "--nlevels",
        type=int,
        default=6,
        help="Number of subdivision levels to create, including level 0",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="In existing OME directory, rebuild higher levels but leave level 0 alone; if this flag is set, the input zarr directory name is still required, but its value is ignored",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=cpu_count(),
        help="Advanced: Number of threads to use for processing. Default is number of CPUs",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index to extract from multi-channel zarr data (0-based indexing)",
    )
    parser.add_argument(
        "--preserve_z",
        action="store_true",
        help="Use Z-preserving scaling (preserve Z resolution)",
    )
    parser.add_argument(
        "--enable_level_smoothing",
        action="store_true",
        help="Enable Gaussian smoothing for higher levels",
    )
    parser.add_argument(
        "--smooth_start_level",
        type=int,
        default=4,
        help="Level to start applying smoothing",
    )
    parser.add_argument(
        "--binary_threshold",
        type=float,
        default=0.5,
        help="Threshold for binary conversion after smoothing",
    )
    parser.add_argument(
        "--keep_continuous",
        action="store_true",
        help="Keep smoothed values continuous instead of binary",
    )

    args = parser.parse_args()

    ozarrdir = Path(args.output_zarr_ome_dir)
    if ozarrdir.suffix != ".zarr":
        print("Name of ouput zarr directory must end with '.zarr'")
        return 1

    izarrdir = args.input_zarr_dir

    nlevels = args.nlevels
    overwrite = args.overwrite
    rebuild_higher_levels = args.rebuild
    num_threads = args.num_threads
    algorithm = args.algorithm
    channel = args.channel
    preserve_z = args.preserve_z
    enable_level_smoothing = args.enable_level_smoothing
    smooth_start_level = args.smooth_start_level

    if args.keep_continuous:
        binary_threshold = None
    else:
        binary_threshold = args.binary_threshold

    print("overwrite", overwrite)

    shift = (0, 0, 0)
    if args.shift is not None:
        shift = parseShift(args.shift)
        if shift is None:
            print("Error parsing shift argument")
            return 1

    slices = (None, None, None)
    if args.window is not None:
        slices = parseSlices(args.window)
        if slices is None:
            print("Error parsing window argument")
            return 1

    chunk_size = args.chunk_size
    compression = args.compression

    dtype = None
    if args.bytes == 1:
        dtype = np.uint8
    elif args.bytes == 2:
        dtype = np.uint16

    if not rebuild_higher_levels:
        if not overwrite:
            if ozarrdir.exists():
                print("Error: Directory", ozarrdir, "already exists")
                return 1

        if ozarrdir.exists():
            print("removing", ozarrdir)
            shutil.rmtree(ozarrdir)

        err = create_ome_dir(ozarrdir)
        if err is not None:
            print("error returned:", err)
            return 1

        if nlevels > 1:
            err = create_ome_headers(ozarrdir, nlevels, preserve_z)
            if err is not None:
                print("error returned:", err)
                return 1

        print("Creating level 0")
        level0dir = ozarrdir / "0"
        if nlevels == 1:
            level0dir = ozarrdir
        print("shift", shift, "slices", slices)
        err = zarr2zarr(
            izarrdir,
            level0dir,
            shift=shift,
            slices=slices,
            chunk_size=chunk_size,
            compression=compression,
            dtype=dtype,
            channel=channel,
        )
        if err is not None:
            print("error returned:", err)
            return 1
    else:
        if nlevels <= 1:
            return 0
        err = create_ome_headers(ozarrdir, nlevels, preserve_z)
        if err is not None:
            print("error returned:", err)
            return 1

    # for each level (1 and beyond):
    existing_level = 0
    for l in range(existing_level, nlevels - 1):
        if preserve_z:
            err = resize_preserve_z(
                ozarrdir,
                l,
                num_threads,
                algorithm,
                enable_level_smoothing,
                smooth_start_level,
                binary_threshold,
            )
        else:
            err = resize(ozarrdir, l, num_threads, algorithm)
        if err is not None:
            print("error returned:", err)
            return 1


if __name__ == "__main__":
    sys.exit(zarr_to_ome_main())
