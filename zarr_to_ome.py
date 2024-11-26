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
from numcodecs.registry import codec_registry
import skimage.transform
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

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
        # print("getValue", key)
        value = self._store[key]
        # print("nv", type(value), len(value))
        if len(value) > 0 and self.compressor is not None:
            for i in range(3):
                try:
                    dc = self.compressor.decode(value)
                    break
                except Exception as e:
                    print("decompression failure try %d: %s"%(i+1, e))
            # print("dc", type(dc), len(dc))
            return dc
        # print("  found", key)
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
                # need to check if key is not in the cache, as it may have been cached
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
        print("Could not parse shift argument '%s'; expected 3 comma-separated numbers"%istr)
        return None
    shift = []
    for sstr in sstrs:
        i = int(sstr)
        if i<3:
            print("shift coordinates must be non-negative")
            return None
        shift.append(i)
    return shift

def parseSlices(istr):
    sstrs = istr.split(",")
    if len(sstrs) != 3:
        print("Could not parse range argument '%s'; expected 3 comma-separated ranges"%istr)
        return (None, None, None)
    slices = []
    for sstr in sstrs:
        if sstr == "":
            slices.append(None)
            continue
        parts = sstr.split(':')
        if len(parts) == 1:
            slices.append(slice(int(parts[0])))
        else:
            iparts = [None if p=="" else int(p) for p in parts]
            if len(iparts)==2:
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

'''
def slice_count(s, maxx):
    mn = s.start
    if mn is None:
        mn = 0
    mn = max(0, mn)
    mx = s.stop
    if mx is None:
        mx = maxx
    mx = min(mx, maxx)
    return mx-mn
'''

def divp1(s, c):
    n = s // c
    if s%c > 0:
        n += 1
    return n

# return None if succeeds, err string if fails
def create_ome_dir(zarrdir):
    # complain if directory already exists
    if zarrdir.exists():
        err = "Directory %s already exists"%zarrdir
        print(err)
        return err

    try:
        zarrdir.mkdir()
    except Exception as e:
        err = "Error while creating %s: %s"%(zarrdir, e)
        print(err)
        return err

def create_ome_headers(zarrdir, nlevels):
    zattrs_dict = {
        "multiscales": [
            {
                "axes": [
                    {
                        "name": "z",
                        "type": "space"
                    },
                    {
                        "name": "y",
                        "type": "space"
                    },
                    {
                        "name": "x",
                        "type": "space"
                    }
                ],
                "datasets": [],
                "name": "/",
                "version": "0.4"
            }
        ]
    }

    dataset_dict = {
        "coordinateTransformations": [
            {
                "scale": [
                ],
                "type": "scale"
            }
        ],
        "path": ""
    }
    
    zgroup_dict = { "zarr_format": 2 }

    datasets = []
    for l in range(nlevels):
        ds = copy.deepcopy(dataset_dict)
        ds["path"] = "%d"%l
        scale = 2.**l
        ds["coordinateTransformations"][0]["scale"] = [scale]*3
        # print(json.dumps(ds, indent=4))
        datasets.append(ds)
    zad = copy.deepcopy(zattrs_dict)
    zad["multiscales"][0]["datasets"] = datasets
    json.dump(zgroup_dict, (zarrdir / ".zgroup").open("w"), indent=4)
    json.dump(zad, (zarrdir / ".zattrs").open("w"), indent=4)

# compression is None or a string.  None means use the
# compression of the input zarr, a string of "none" or "" means 
# no compression; otherwise the string gives the name of the compression
# scheme.
# If chunk_size is None, the input zarr's chunking is used, otherwise
# chunks of (chunk_size, chunk_size, chunk_size) are used.
# If dtype is None, the input zarr's dtype is used.  Otherwise the
# given dtype is used.  Note that the only supported conversion
# is from uint16 to uint8.
def zarr2zarr(izarrdir, ozarrdir, shift=(0,0,0), slices=(None,None,None), chunk_size=None, compression=None, dtype=None):
    # user gives shift and slices in x,y,z order, but internally
    # we use z,y,x order
    shift = (shift[2], shift[1], shift[0])
    slices = (slices[2], slices[1], slices[0])
    # TODO: slice_start >= 0 for all slices
    if not all([(slice_start(s)>=0) for s in slices]):
        err = "All window starting coordinates must be >= 0"
        print(err)
        return err
    if not all([slice_step_is_1(s) for s in slices]):
        err = "All window steps must be 1"
        print(err)
        return err
    izarr = zarr.open(izarrdir, mode="r")
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
            err = "Can't convert %s to %s"%(str(izarr.dtype), str(dtype))
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

    # TODO: allow larger max_size
    # TODO: store decompressed chunks in cache
    store = izarr.store
    # cstore = zarr.LRUStoreCache(store, max_size=2**32)
    cstore = DecompressedLRUCache(store, max_size=2**32)
    czarr = zarr.open(cstore, mode="r")
    cstore.transferCompressor(czarr)
    # czarr = izarr
    ishape = izarr.shape
    # is0 and is1 are the min and max of the input zarr, shifted
    # to the output zarr's grid
    is0 = [shift[i] for i in range(3)]
    is1 = [ishape[i]+shift[i] for i in range(3)]
    # ozarr covers the range 0,0,0 to stop-point of given window 
    # (or max of shifted izarr, if no window given)
    oshape = [slice_stop(slices[i], is1[i]) for i in range(3)]
    # os0 and os1 are the min and max of the region in
    # ozarr that should be written to, based on the given
    # window (if none given, use 0,0,0 and max of shifted izarr)
    os0 = [slice_start(slices[i], 0) for i in range(3)]
    # os1 = [slice_stop(slices[i], is1[i]) for i in range(3)]
    os1 = [oshape[i] for i in range(3)]
    if not all([(os1[i] > os0[i]) for i in range(3)]):
        err = "Computed conflicting output grid min %s and max %s"%(str(os0), str(os1))
        print(err)
        return err
    # cs0 and cs1 are the min and max of the part of the input grid
    # that should be copied to the output grid, 
    # in shifted (output-grid) coordinates
    cs0 = [max(os0[i], is0[i]) for i in range(3)]
    cs1 = [min(os1[i], is1[i]) for i in range(3)]
    '''
    # Note that the code below is wrong; we want the intersection
    # between is0 to is1,  and 0,0,0 to oshape
    cs0 = [max(slice_start(slices[i]), is0[i]) for i in range(3)]
    # TODO: This won't work!!!
    cs1 = [min(slice_stop(slices[i]),  is1[i]) for i in range(3)]
    # TODO: make sure cs1 > 0
    # oshape = [shift[i]+ishape[i] for i in range(3)]
    '''

    store = zarr.NestedDirectoryStore(ozarrdir)
    ozarr = zarr.open(
            store=store, 
            shape=oshape, 
            chunks=chunk_sizes,
            dtype = dtype,
            write_empty_chunks=False,
            fill_value=0,
            compressor=compressor,
            mode='w', 
            )

    # chunk0s = [shift[i] // chunk_sizes[i] for i in range(3)]
    # nchunks = [divp1(oshape[i], chunk_sizes[i]) for i in range(3)]
    chunk0s = [cs0[i] // chunk_sizes[i] for i in range(3)]
    chunk1s = [cs1[i] // chunk_sizes[i] for i in range(3)]
    print("chunk_sizes", chunk_sizes)
    # print("chunk0s", chunk0s)
    # print("nchunks", nchunks)
    ci = [0,0,0]
    # for ci[0] in range(chunk0s[0], nchunks[0]):
    for ci[0] in range(chunk0s[0], chunk1s[0]):
        print("doing", ci[0], "to", chunk1s[0])
        for ci[1] in range(chunk0s[1], chunk1s[1]):
            for ci[2] in range(chunk0s[2], chunk1s[2]):
                o0 = [max(cs0[i], chunk_sizes[i]*ci[i]) for i in range(3)]
                o1 = [min(cs1[i], chunk_sizes[i]*(1+ci[i])) for i in range(3)]
                i0 = [o0[i]-shift[i] for i in range(3)]
                i1 = [o1[i]-shift[i] for i in range(3)]
                '''
                o0 = [chunk_sizes[i]*ci[i] for i in range(3)]
                o1 = [o0[i]+chunk_sizes[i] for i in range(3)]
                i0 = [o0[i]-shift[i] for i in range(3)]
                s0 = [max(0,-i0[i]) for i in range(3)]
                o0 = [o0[i]+s0[i] for i in range(3)]
                i0 = [o0[i]-shift[i] for i in range(3)]
                i1 = [o1[i]-shift[i] for i in range(3)]
                '''
                # print("o0", o0)
                # print("o1", o1)
                # print("i0", i0)
                # print("i1", i1)
                ozarr[o0[0]:o1[0], o0[1]:o1[1], o0[2]:o1[2]] = czarr[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] // divisor


def process_chunk(args):
    idata, odata, z, y, x, cz, cy, cx, algorithm = args
    ibuf = idata[2*z*cz:(2*z*cz+2*cz),
                 2*y*cy:(2*y*cy+2*cy),
                 2*x*cx:(2*x*cx+2*cx)]
    if np.max(ibuf) == 0:
        return  # Skip if the block is empty to save computation

    # pad ibuf to even in all directions
    ibs = ibuf.shape
    pad = (ibs[0]%2, ibs[1]%2, ibs[2]%2)
    if any(pad):
        ibuf = np.pad(ibuf, 
                      ((0,pad[0]),(0,pad[1]),(0,pad[2])), 
                      mode="symmetric")

    # algorithms:
    if algorithm == "nearest":
        obuf = ibuf[::2, ::2, ::2]
    elif algorithm == "gaussian":
        obuf = np.round(skimage.transform.rescale(ibuf, .5, preserve_range=True))
    elif algorithm == "mean":
        obuf = np.round(skimage.transform.downscale_local_mean(ibuf, (2,2,2)))
    else:
        raise ValueError(f"algorithm {algorithm} not valid")

    odata[z*cz:(z*cz+cz),
          y*cy:(y*cy+cy),
          x*cx:(x*cx+cx)] = np.round(obuf)

def resize(zarrdir, old_level, num_threads, algorithm="mean"):
    idir = zarrdir / ("%d"%old_level)
    if not idir.exists():
        err = f"input directory {idir} does not exist"
        print(err)
        return err
    
    odir = zarrdir / ("%d"%(old_level+1))
    idata = zarr.open(idir, mode="r")
    print("Creating level",old_level+1,"  input array shape", idata.shape, " algorithm", algorithm)

    cz, cy, cx = idata.chunks
    sz, sy, sx = idata.shape
    store = zarr.NestedDirectoryStore(odir)
    odata = zarr.open(
            store=store,
            shape=(divp1(sz,2), divp1(sy,2), divp1(sx,2)),
            chunks=idata.chunks,
            dtype=idata.dtype,
            write_empty_chunks=False,
            fill_value=0,
            compressor=idata.compressor,
            mode='w',
            )

    # Prepare tasks
    tasks = [(idata, odata, z, y, x, cz, cy, cx, algorithm) for z in range(divp1(sz, 2*cz))
                                                             for y in range(divp1(sy, 2*cy))
                                                             for x in range(divp1(sx, 2*cx))]

    # Use ThreadPoolExecutor to process blocks in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(process_chunk, tasks), total=len(tasks)))

    print("Processing complete")

def main():
    # parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Create OME/Zarr data store from an existing zarr data store")
    parser.add_argument(
            "input_zarr_dir", 
            help="Name of zarr store directory")
    parser.add_argument(
            "output_zarr_ome_dir", 
            help="Name of directory that will contain OME/zarr datastore")
    parser.add_argument(
            "--nlevels", 
            type=int, 
            default=6, 
            help="Number of subdivision levels to create, including level 0")
    parser.add_argument(
            "--overwrite", 
            action="store_true", 
            help="Overwrite the output directory, if it already exists")
    parser.add_argument(
            "--num_threads", 
            type=int, 
            default=cpu_count(), 
            help="Advanced: Number of threads to use for processing. Default is number of CPUs")
    parser.add_argument(
            "--algorithm",
            choices=['mean', 'gaussian', 'nearest'],
            default="mean",
            help="Advanced: algorithm used to sub-sample the data")
    parser.add_argument(
            "--shift", 
            help="Advanced: shift input data set, relative to output data set.  Example (in xyz order): 2000,3000,5500")
    parser.add_argument(
            "--window", 
            help="Advanced: output only a subset of the data.  Example (in xyz order, and based on the output-data grid): 2500:3000,1500:4000,500:600")
    
    parser.add_argument(
            "--chunk_size", 
            type=int,
            help="Size of chunk; if not given, will be same as input zarr")
    parser.add_argument(
            "--compression", 
            help="Compression algorithm ('blosc', 'none', ...); if not given, will be same as input zarr; if set to 'none', no compression will be used")
    parser.add_argument(
            "--bytes",
            type=int,
            default=0,
            help="number of bytes per pixel in output; if not given, will be same as input zarr")

    args = parser.parse_args()
    
    ozarrdir = Path(args.output_zarr_ome_dir)
    if ozarrdir.suffix != ".zarr":
        print("Name of ouput zarr directory must end with '.zarr'")
        return 1
    
    izarrdir = Path(args.input_zarr_dir)
    if not izarrdir.exists():
        print("Input zarr directory",izarrdir,"does not exist")
        return 1

    nlevels = args.nlevels
    overwrite = args.overwrite
    num_threads = args.num_threads
    algorithm = args.algorithm
    print("overwrite", overwrite)

    shift = (0,0,0)
    if args.shift is not None:
        shift = parseShift(args.shift)
        if shift is None:
            print("Error parsing shift argument")
            return 1
    
    slices = (None,None,None)
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
    
    if not overwrite:
        if ozarrdir.exists():
            print("Error: Directory",ozarrdir,"already exists")
            return(1)
    
    if ozarrdir.exists():
        print("removing", ozarrdir)
        shutil.rmtree(ozarrdir)
    
    err = create_ome_dir(ozarrdir)
    if err is not None:
        print("error returned:", err)
        return 1
    
    err = create_ome_headers(ozarrdir, nlevels)
    if err is not None:
        print("error returned:", err)
        return 1
    
    print("Creating level 0")
    print("shift", shift, "slices", slices)
    err = zarr2zarr(izarrdir, ozarrdir/"0", shift=shift, slices=slices, chunk_size=chunk_size, compression=compression, dtype=dtype)
    if err is not None:
        print("error returned:", err)
        return 1
    
    # for each level (1 and beyond):
    existing_level = 0
    for l in range(existing_level, nlevels-1):
        err = resize(ozarrdir, l, num_threads, algorithm)
        if err is not None:
            print("error returned:", err)
            return 1

if __name__ == '__main__':
    sys.exit(main())
