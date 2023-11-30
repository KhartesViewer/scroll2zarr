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
import skimage.transform

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

# for each level (1 and beyond):
# resize(zarrdir, old level, resize algo)

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

def tifs2zarr(tiffdir, zarrdir, chunk_size, slices=None, maxgb=None):
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
    
    tiff0 = tifffile.imread(inttiffs[minz])
    ny0, nx0 = tiff0.shape
    dt0 = tiff0.dtype
    print("tiff size", nx0, ny0, "z range", minz, maxz)

    cx = nx0
    cy = ny0
    if xslice is not None:
        cx = slice_count(xslice, nx0)
    if yslice is not None:
        cy = slice_count(yslice, ny0)
    print("cx,cy,cz",cx,cy,cz)
    
    store = zarr.NestedDirectoryStore(zarrdir)
    tzarr = zarr.open(
            store=store, 
            shape=(cz, cy, cx), 
            chunks=(chunk_size, chunk_size, chunk_size),
            dtype = tiff0.dtype,
            write_empty_chunks=False,
            fill_value=0,
            compressor=None,
            mode='w', 
            )
    
    # allow buf that is smaller than chunk size
    bufnz = chunk_size
    if maxgb is not None:
        mxz = int((maxgb*10**9)/(cx*cy*dt0.itemsize))
        bufnz = min(bufnz, mxz)
    buf = np.zeros((bufnz, cy, cx), dtype=dt0)
    prev_zc = -1
    prev_zb = -1
    cur_bufz = -1
    for itiff in itiffs:
        z = itiff-z0
        tiffname = inttiffs[itiff]
        print("reading",itiff)
        tarr = tifffile.imread(tiffname)
        # tzarr[itiff,:,:] = tarr
        ny, nx = tarr.shape
        if nx != nx0 or ny != ny0:
            print("File %s is the wrong shape (%d, %d); expected %d, %d"%(tiffname,nx,ny,nx0,ny0))
            continue
        if xslice is not None and yslice is not None:
            tarr = tarr[yslice, xslice]
        cur_zc = z // chunk_size
        cur_zb = (z-cur_zc*chunk_size) // bufnz
        if cur_zc != prev_zc:
            if prev_zc >= 0:
                zs = prev_zc*chunk_size+prev_zb*bufnz
                print("writing (zc)", zs, zs+bufnz)
                tzarr[zs:zs+bufnz,:,:] = buf
                buf[:,:,:] = 0
            prev_zc = cur_zc
            prev_zb = -1
        elif cur_zb != prev_zb:
            if prev_zb >= 0:
                zs = cur_zc*chunk_size+prev_zb*bufnz
                zend = min(zs+bufnz, chunk_size)
                print("writing (zb)", zs, zend)
                tzarr[zs:zend,:,:] = buf
                buf[:,:,:] = 0
            prev_zb = cur_zb
        cur_bufz = z-cur_zc*chunk_size-cur_zb*bufnz
        buf[cur_bufz,:,:] = tarr
    
    if prev_zc >= 0:
        zs = prev_zc*chunk_size
        if prev_zb > 0:
            zs += prev_zb*bufnz
        print("writing (end)", zs, zs+bufnz)
        tzarr[zs:zs+bufnz,:,:] = buf[0:(1+cur_bufz)]

def divp1(s, c):
    n = s // c
    if s%c > 0:
        n += 1
    return n

def resize(zarrdir, old_level, algorithm="mean"):
    idir = zarrdir / ("%d"%old_level)
    if not idir.exists():
        err = "input directory %s does not exist" % idir
        print(err)
        return(err)
    odir = zarrdir / ("%d"%(old_level+1))
    # print(zarrdir, idir, odir)
    
    idata = zarr.open(idir, mode="r")
    
    print(idata.chunks, idata.shape)
    
    # order is z,y,x
    
    cz = idata.chunks[0]
    cy = idata.chunks[1]
    cx = idata.chunks[2]
    
    sz = idata.shape[0]
    sy = idata.shape[1]
    sx = idata.shape[2]
    
    store = zarr.NestedDirectoryStore(odir)
    odata = zarr.open(
            store=store,
            shape=(divp1(sz,2), divp1(sy,2), divp1(sx,2)),
            chunks=idata.chunks,
            dtype=idata.dtype,
            write_empty_chunks=False,
            fill_value=0,
            compressor=None,
            mode='w',
            )
    
    # 2*chunk size because we want number of blocks after rescale
    nz = divp1(sz, 2*cz)
    ny = divp1(sy, 2*cy)
    nx = divp1(sx, 2*cx)
    
    print("nzyx", nz,ny,nx)
    ibuf = np.zeros((2*cz,2*cy,2*cx), dtype=idata.dtype)
    for z in range(nz):
        print("z", z)
        for y in range(ny):
            for x in range(nx):
                ibuf = idata[
                        2*z*cz:(2*z*cz+2*cz),
                        2*y*cy:(2*y*cy+2*cy),
                        2*x*cx:(2*x*cx+2*cx)]
                if np.max(ibuf) == 0:
                    continue
                # pad ibuf to even in all directions
                ibs = ibuf.shape
                pad = (ibs[0]%2, ibs[1]%2, ibs[2]%2)
                if any(pad):
                    ibuf = np.pad(ibuf, 
                                  ((0,pad[0]),(0,pad[1]),(0,pad[2])), 
                                  mode="symmetric")
                    print("padded",ibs,"to",ibuf.shape)
                # algorithms:
                if algorithm == "nearest":
                    obuf = ibuf[::2, ::2, ::2]
                elif algorithm == "gaussian":
                    obuf = np.round(
                        skimage.transform.rescale(
                            ibuf, .5, preserve_range=True))
                elif algorithm == "mean":
                    obuf = np.round(
                        skimage.transform.downscale_local_mean(
                            ibuf, (2,2,2)))
                else:
                    err = "algorithm %s not valid"%algorithm
                    print(err)
                    return err
                print(np.max(obuf), x, y, z)
                odata[ z*cz:(z*cz+cz),
                       y*cy:(y*cy+cy),
                       x*cx:(x*cx+cx)] = np.round(obuf)


def main():
    # parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Create OME/Zarr data store from a set of TIFF files")
    parser.add_argument(
            "input_tiff_dir", 
            help="Directory containing tiff files")
    parser.add_argument(
            "output_zarr_ome_dir", 
            help="Name of directory that will contain OME/zarr datastore")
    parser.add_argument(
            "--chunk_size", 
            type=int, 
            default=128, 
            help="Size of chunk")
    parser.add_argument(
            "--nlevels", 
            type=int, 
            default=6, 
            help="Number of subdivision levels to create, including level 0")
    parser.add_argument(
            "--max_gb", 
            type=float, 
            default=None, 
            help="Maximum amount of memory (in Gbytes) to use; None means no limit")
    parser.add_argument(
            "--zarr_only", 
            action="store_true", 
            help="Create a simple Zarr data store instead of an OME/Zarr hierarchy")
    parser.add_argument(
            "--overwrite", 
            action="store_true", 
            # default=False,
            help="Overwrite the output directory, if it already exists")
    parser.add_argument(
            "--algorithm",
            choices=['mean', 'gaussian', 'nearest'],
            default="mean",
            help="Advanced: algorithm used to sub-sample the data")
    parser.add_argument(
            "--ranges", 
            help="Advanced: output only a subset of the data.  Example (in xyz order): 2500:3000,1500:4000,500:600")
    parser.add_argument(
            "--first_new_level", 
            type=int, 
            default=None, 
            help="Advanced: If some subdivision levels already exist, create new levels, starting with this one")

    args = parser.parse_args()
    
    zarrdir = Path(args.output_zarr_ome_dir)
    if zarrdir.suffix != ".zarr":
        print("Name of ouput zarr directory must end with '.zarr'")
        return 1
    
    tiffdir = Path(args.input_tiff_dir)
    chunk_size = args.chunk_size
    nlevels = args.nlevels
    maxgb = args.max_gb
    zarr_only = args.zarr_only
    overwrite = args.overwrite
    algorithm = args.algorithm
    print("overwrite", overwrite)
    first_new_level = args.first_new_level
    if first_new_level is not None and first_new_level < 1:
        print("first_new_level must be at least 1")
    
    slices = None
    if args.ranges is not None:
        slices = parseSlices(args.ranges)
        if slices is None:
            print("Error parsing ranges argument")
            return 1
    
    print("slices", slices)
    
    if overwrite and (first_new_level is None or zarr_only):
        if zarrdir.exists():
            print("removing", zarrdir)
            shutil.rmtree(zarrdir)
    
    # tifs2zarr(tiffdir, zarrdir, chunk_size, slices=slices, maxgb=maxgb)
    
    if zarr_only:
        err = tifs2zarr(tiffdir, zarrdir, chunk_size, slices=slices, maxgb=maxgb)
        if err is not None:
            print("error returned:", err)
        return 1
    
    if first_new_level is None:
        err = create_ome_dir(zarrdir)
        if err is not None:
            print("error returned:", err)
            return 1
    
    err = create_ome_headers(zarrdir, nlevels)
    if err is not None:
        print("error returned:", err)
        return 1
    
    if first_new_level is None:
        err = tifs2zarr(tiffdir, zarrdir/"0", chunk_size, slices=slices, maxgb=maxgb)
        if err is not None:
            print("error returned:", err)
            return 1
    
    # for each level (1 and beyond):
    existing_level = 0
    if first_new_level is not None:
        existing_level = first_new_level-1
    for l in range(existing_level, nlevels-1):
        err = resize(zarrdir, l, algorithm)
        if err is not None:
            print("error returned:", err)
            return 1

if __name__ == '__main__':
    sys.exit(main())
