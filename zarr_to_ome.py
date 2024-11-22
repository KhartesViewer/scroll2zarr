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

'''
# tiffdir = Path(r"C:\Vesuvius\scroll 1 2000-2030")
# zarrdir = Path(r"C:\Vesuvius\test.zarr")
# tiffdir = Path(r"H:\Vesuvius\Scroll1.volpkg\volumes_masked\20230205180739")
# tiffdir = Path(r"H:\Vesuvius\zarr_tests\masked_subset")
# zarrdir = Path(r"H:\Vesuvius\zarr_tests\testzo.zarr")
# zarrdir = Path(r"H:\Vesuvius\testzc.zarr")

# tif files, ome dir
# set chunk_size
chunk_size = 128
# slices = (None, None, slice(1975,2010))
# slices = (slice(2000,2500), slice(2000,2512), slice(1975,2010))
slices = (slice(2000,2500), slice(2000,2512), slice(1975,2005))
maxgb = None
nlevels = 6
zarr_only = False
first_new_level = 0
# maxgb = .0036
'''

# create ome dir, .zattrs, .zgroup
# (don't need to know output array dimensions, just number of levels,
# possibly unit/dimension info)
# create_ome_dir(zarrdir, nlevels)
# quit if dir already exists

# tifs2zarr(tiffdir, zarrdir+"/0", chunk_size, range(optional))

def parseCorner(istr):
    sstrs = istr.split(",")
    if len(sstrs) != 3:
        print("Could not parse corner argument '%s'; expected 3 comma-separated numbers"%istr)
        return None
    corner = []
    for sstr in sstrs:
        i = int(sstr)
        if i<3:
            print("corner coordinates must be non-negative")
            return None
        corner.append(i)
    return corner

def parseSlices(istr):
    sstrs = istr.split(",")
    if len(sstrs) != 3:
        print("Could not parse ranges argument '%s'; expected 3 comma-separated ranges"%istr)
        return None
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

def slice_step_is_1(s):
    if s is None:
        return True
    if s.step is None:
        return True
    if s.step == 1:
        return True
    return False

def slice_start(s):
    if s.start is None:
        return 0
    return s.start

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

def divp1(s, c):
    n = s // c
    if s%c > 0:
        n += 1
    return n

# compression is None or a string.  None means use the
# compression of the input zarr, a string of "none" or "" means 
# no compression; otherwise the string gives the name of the compression
# scheme.
# If chunk_size is None, the input zarr's chunking is used, otherwise
# chunks of (chunk_size, chunk_size, chunk_size) are used.
# If dtype is None, the input zarr's dtype is used.  Otherwise the
# given dtype is used.  Note that the only supported conversion
# is from uint16 to uint8.
def zarr2zarr(izarrdir, ozarrdir, corner=(0,0,0), chunk_size=None, compression=None, dtype=None):
    # user gives corner as x,y,z, but internally
    # we use z,y,x
    corner = (corner[2], corner[1], corner[0])
    izarr = zarr.open(izarrdir, mode="r")
    store = izarr.store
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
            print("Can't convert",izarr.dtype,"to",dtype)
            return

    compressor = izarr.compressor
    if compression is not None:
        if compression == "none" or compression == "":
            compressor = None
        else:
            codec_cls = codec_registry[compression]
            compressor = codec_cls()
            # if compression_opts is dict:
            # compressor = codec_cls(**compression_opts)
    czarr = zarr.LRUStoreCache(store, max_size=2**32)
    ishape = izarr.shape
    oshape = [corner[i]+ishape[i] for i in range(3)]
    nchunks = [divp1(oshape[i], chunk_sizes[i]) for i in range(3)]

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

    chunk0s = [corner[i] // chunk_sizes[i] for i in range(3)]
    print("chunk_sizes", chunk_sizes)
    # print("chunk0s", chunk0s)
    # print("nchunks", nchunks)
    ci = [0,0,0]
    for ci[0] in range(chunk0s[0], nchunks[0]):
        print("doing", ci[0], "of", nchunks[0])
        for ci[1] in range(chunk0s[1], nchunks[1]):
            for ci[2] in range(chunk0s[2], nchunks[2]):
                o0 = [chunk_sizes[i]*ci[i] for i in range(3)]
                o1 = [o0[i]+chunk_sizes[i] for i in range(3)]
                i0 = [o0[i]-corner[i] for i in range(3)]
                s0 = [max(0,-i0[i]) for i in range(3)]
                o0 = [o0[i]+s0[i] for i in range(3)]
                i0 = [o0[i]-corner[i] for i in range(3)]
                i1 = [o1[i]-corner[i] for i in range(3)]
                # print("o0", o0)
                # print("o1", o1)
                # print("i0", i0)
                # print("i1", i1)
                ozarr[o0[0]:o1[0], o0[1]:o1[1], o0[2]:o1[2]] = izarr[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] // divisor


def tifs2zarr(tiffdir, zarrdir, chunk_size, obytes=0, slices=None, maxgb=None):
    if slices is None:
        xslice = yslice = zslice = None
    else:
        xslice, yslice, zslice = slices
        if not all([slice_step_is_1(s) for s in slices]):
            err = "All slice steps must be 1 in slices"
            print(err)
            return err
    # Note this is a generator, not a list
    tiffs = tiffdir.glob("*.tif")
    rec = re.compile(r'([0-9]+)\.\w+$')
    # rec = re.compile(r'[0-9]+$')
    inttiffs = {}
    for tiff in tiffs:
        tname = tiff.name
        match = rec.match(tname)
        if match is None:
            continue
        # Look for last match (closest to end of file name)
        # ds = match[-1]
        ds = match.group(1)
        itiff = int(ds)
        if itiff in inttiffs:
            err = "File %s: tiff id %d already used"%(tname,itiff)
            print(err)
            return err
        inttiffs[itiff] = tiff
    if len(inttiffs) == 0:
        err = "No tiffs found"
        print(err)
        return err
    
    itiffs = list(inttiffs.keys())
    itiffs.sort()
    z0 = 0
    if zslice is not None:
        maxz = itiffs[-1]+1
        valid_zs = range(maxz)[zslice]
        itiffs = list(filter(lambda z: z in valid_zs, itiffs))
        # z0 = itiffs[0]
        if zslice.start is None:
            z0 = 0
        else:
            z0 = zslice.start
    
    # for testing
    # itiffs = itiffs[2048:2048+256]
    
    minz = itiffs[0]
    maxz = itiffs[-1]
    cz = maxz-z0+1
    
    try:
        tiff0 = tifffile.imread(inttiffs[minz])
    except Exception as e:
        err = "Error reading %s: %s"%(inttiffs[minz],e)
        print(err)
        return err
    ny0, nx0 = tiff0.shape
    dt0 = tiff0.dtype
    otype = tiff0.dtype
    divisor = 1
    if obytes == 1 and dt0 == np.uint16:
        print("Converting from uint16 in input to uint8 in output")
        otype = np.uint8
        divisor = 256
    elif obytes != 0 and dt0.itemsize != obytes:
        err = "Cannot perform pixel conversion from %s to %d bytes"%(dt0, obytes)
        print(err)
        return err
    else:
        print("Byte conversion: none")
    print("tiff size", nx0, ny0, "z range", minz, maxz)

    cx = nx0
    cy = ny0
    x0 = 0
    y0 = 0
    if xslice is not None:
        cx = slice_count(xslice, nx0)
        x0 = slice_start(xslice)
    if yslice is not None:
        cy = slice_count(yslice, ny0)
        y0 = slice_start(yslice)
    print("cx,cy,cz",cx,cy,cz)
    print("x0,y0,z0",x0,y0,z0)
    
    store = zarr.NestedDirectoryStore(zarrdir)
    tzarr = zarr.open(
            store=store, 
            shape=(cz, cy, cx), 
            chunks=(chunk_size, chunk_size, chunk_size),
            dtype = otype,
            write_empty_chunks=False,
            fill_value=0,
            compressor=None,
            mode='w', 
            )

    # nb of chunks in y direction that fit inside of max_gb
    chy = cy // chunk_size + 1
    if maxgb is not None:
        maxy = int((maxgb*10**9)/(cx*chunk_size*dt0.itemsize))
        chy = maxy // chunk_size
        chy = max(1, chy)

    # nb of y chunk groups
    ncgy = cy // (chunk_size*chy) + 1
    print("chy, ncgy", chy, ncgy)
    buf = np.zeros((chunk_size, min(cy, chy*chunk_size), cx), dtype=dt0)
    for icy in range(ncgy):
        ys = icy*chy*chunk_size
        ye = ys+chy*chunk_size
        ye = min(ye, cy)
        if ye == ys:
            break
        prev_zc = -1
        for itiff in itiffs:
            z = itiff-z0
            tiffname = inttiffs[itiff]
            try:
                print("reading",itiff,"     ", end='\r')
                # print("reading",itiff)
                tarr = tifffile.imread(tiffname)
            except Exception as e:
                print("\nError reading",tiffname,":",e)
                # If reading fails (file missing or deformed)
                tarr = np.zeros((ny, nx), dtype=dt0)
            # print("done reading",itiff, end='\r')
            # tzarr[itiff,:,:] = tarr
            ny, nx = tarr.shape
            if nx != nx0 or ny != ny0:
                print("\nFile %s is the wrong shape (%d, %d); expected %d, %d"%(tiffname,nx,ny,nx0,ny0))
                continue
            if xslice is not None and yslice is not None:
                tarr = tarr[yslice, xslice]
            cur_zc = z // chunk_size
            if cur_zc != prev_zc:
                if prev_zc >= 0:
                    zs = prev_zc*chunk_size
                    ze = zs+chunk_size
                    if ncgy == 1:
                        print("\nwriting, z range %d,%d"%(zs+z0, ze+z0))
                    else:
                        print("\nwriting, z range %d,%d  y range %d,%d"%(zs+z0, ze+z0, ys+y0, ye+y0))
                    tzarr[zs:z,ys:ye,:] = buf[:ze-zs,:ye-ys,:]
                    buf[:,:,:] = 0
                prev_zc = cur_zc
            cur_bufz = z-cur_zc*chunk_size
            # print("cur_bufzk,ye,ys", cur_bufz,ye,ys)
            buf[cur_bufz,:ye-ys,:] = tarr[ys:ye,:] // divisor
        
        if prev_zc >= 0:
            zs = prev_zc*chunk_size
            ze = zs+chunk_size
            ze = min(itiffs[-1]+1-z0, ze)
            if ze > zs:
                if ncgy == 1:
                    print("\nwriting, z range %d,%d"%(zs+z0, ze+z0))
                else:
                    print("\nwriting, z range %d,%d  y range %d,%d"%(zs+z0, ze+z0, ys+y0, ye+y0))
                # print("\nwriting (end)", zs, ze)
                # tzarr[zs:zs+bufnz,:,:] = buf[0:(1+cur_bufz)]
                tzarr[zs:ze,ys:ye,:] = buf[:ze-zs,:ye-ys,:] // divisor
            else:
                print("\n(end)")
        buf[:,:,:] = 0

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
            "--corner", 
            help="Advanced: starting corner of input data set, relative to origin of output data set.  Example (in xyz order): 2000,3000,5500")
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
    
    zarrdir = Path(args.output_zarr_ome_dir)
    if zarrdir.suffix != ".zarr":
        print("Name of ouput zarr directory must end with '.zarr'")
        return 1
    
    tiffdir = Path(args.input_zarr_dir)
    if not tiffdir.exists():
        print("Input zarr directory",tiffdir,"does not exist")
        return 1

    nlevels = args.nlevels
    overwrite = args.overwrite
    num_threads = args.num_threads
    algorithm = args.algorithm
    print("overwrite", overwrite)

    corner = (0,0,0)
    if args.corner is not None:
        corner = parseCorner(args.corner)
        if corner is None:
            print("Error parsing corner argument")
            return 1

    chunk_size = args.chunk_size
    compression = args.compression

    dtype = None
    if args.bytes == 1:
        dtype = np.uint8
    elif args.bytes == 2:
        dtype = np.uint16
    
    if not overwrite:
        if zarrdir.exists():
            print("Error: Directory",zarrdir,"already exists")
            return(1)
    
    if zarrdir.exists():
        print("removing", zarrdir)
        shutil.rmtree(zarrdir)
    
    err = create_ome_dir(zarrdir)
    if err is not None:
        print("error returned:", err)
        return 1
    
    err = create_ome_headers(zarrdir, nlevels)
    if err is not None:
        print("error returned:", err)
        return 1
    
    print("Creating level 0")
    print("corner", corner)
    err = zarr2zarr(tiffdir, zarrdir/"0", corner=corner, chunk_size=chunk_size, compression=compression, dtype=dtype)
    if err is not None:
        print("error returned:", err)
        return 1
    
    # for each level (1 and beyond):
    existing_level = 0
    for l in range(existing_level, nlevels-1):
        err = resize(zarrdir, l, num_threads, algorithm)
        if err is not None:
            print("error returned:", err)
            return 1

if __name__ == '__main__':
    sys.exit(main())
