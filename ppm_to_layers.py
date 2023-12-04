import sys
import math
import argparse
from pathlib import Path
import numpy as np
import zarr
import tifffile
from scipy.interpolate import RegularGridInterpolator
from ppm import Ppm


# From https://neurostars.org/t/trilinear-interpolation-in-python/18019
def trilinear(fxyz, data):
    '''
    xyz: array with coordinates inside data
    data: 3d volume
    returns: interpolated data values at coordinates
    '''
    ijk = fxyz.astype(np.int32)
    i, j, k = ijk[...,0], ijk[...,1], ijk[...,2]
    V000 = data[ i   , j   ,  k   ]
    V100 = data[(i+1), j   ,  k   ]
    V010 = data[ i   ,(j+1),  k   ]
    V001 = data[ i   , j   , (k+1)]
    V101 = data[(i+1), j   , (k+1)]
    V011 = data[ i   ,(j+1), (k+1)]
    V110 = data[(i+1),(j+1),  k   ]
    V111 = data[(i+1),(j+1), (k+1)]
    xyz = fxyz - ijk
    x, y, z = xyz[...,0], xyz[...,1], xyz[...,2]
    Vxyz = (V000 * (1 - x)*(1 - y)*(1 - z)
            + V100 * x * (1 - y) * (1 - z) +
            + V010 * (1 - x) * y * (1 - z) +
            + V001 * (1 - x) * (1 - y) * z +
            + V101 * x * (1 - y) * z +
            + V011 * (1 - x) * y * z +
            + V110 * x * y * (1 - z) +
            + V111 * x * y * z)
    return Vxyz

# TODO: try memory mapped directory store: 
# https://github.com/zarr-developers/zarr-python/pull/377

class KhartesLRUCache(zarr.storage.LRUStoreCache):
    def __init__(self, store, max_size):
        super().__init__(store, max_size)
        self.prev = {}

    def __getitem__old(self, key):
        print("get item", key)
        return super().__getitem__(key)

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
            value = self._store[key]
            # print("cache miss", key)
            if key in self.prev:
                print(self.prev[key], key)
            self.prev[key] = self.prev.get(key, 0) + 1
            with self._mutex:
                self.misses += 1
                # need to check if key is not in the cache, as it may have been cached
                # while we were retrieving the value from the store
                if key not in self._values_cache:
                    self._cache_value(key, value)

        return value

'''
def get_output_paths():
    ofl = Path(r'D:\Vesuvius\Projects\ppmtest\layers128ta')
    sfl = Path(r'D:\Vesuvius\Projects\ppmtest\layers128ta.tif')
    # ofl = Path(r'D:\Vesuvius\Projects\ppmtest\layers64')
    # ofl = Path(r"D:\Vesuvius\Projects\Bigseg\layers128t100")
    return ofl, sfl
'''

def create_output_dir(path):
    if not path.exists():
        path.mkdir()
    return True

'''
# ppmfl, zdir, step, maxgb, layer_half_width, stack_half_width
def get_params():
    step = 100
    # step = 200

    maxgb = 8 # after 6200; count 3950
    maxgb = 20 # None; count 3950, 0
    maxgb = 10 # after 6200; count 3950, 648
    maxgb = 15 # None; count 3950, 0
    maxgb = 5 # after 6200; count 3950, 648 but after switching u,v: None!

    ppmfl = Path(r'D:\Vesuvius\Projects\ppmtest\1846.ppm')
    # ppmfl = Path(r"D:\Vesuvius\Projects\Bigseg\20231022170900.ppm")
    # zdir = Path(r'H:\Vesuvius\zarr_tests\scroll1m.zarr\0')
    zdir = Path(r'D:\Vesuvius\scroll1_128.zarr\0')
    # zdir = Path(r'D:\Vesuvius\scroll1_64.zarr\0')

    layer_half_width = 32
    stack_half_width = 12
    return ppmfl, zdir, step, maxgb, layer_half_width, stack_half_width
'''

# xyzs, normals, normals_valid
def get_ppm_data(ppmfl):
    ppm = Ppm.loadPpm(ppmfl)
    ppm.loadData()
    xyzs = ppm.ijks.astype(np.float32)
    print("xyzs", xyzs.shape, xyzs.dtype)
    normals = ppm.normals.astype(np.float32)
    normals_valid = (normals != 0).any(axis=2)
    zmin = np.min(xyzs[normals_valid, 2])
    zmax = np.max(xyzs[normals_valid,2])
    print("zmin, zmax", zmin, zmax)
    return xyzs, normals, normals_valid

def get_zarr_data(zdir, maxgb):
    z_uncached_data = zarr.open(zdir, "r")
    if not hasattr(z_uncached_data, "shape"):
        print(zdir,"is not a valid zarr store")
        print("Perhaps it is an OME-Zarr store?")
        zdir = zdir / '0'
        print("Trying",zdir)
        z_uncached_data = zarr.open(zdir, "r")
    zdata = zarr.open(KhartesLRUCache(z_uncached_data.store, max_size=maxgb*2**30), mode="r")
    # TODO: detect if zdata is a group instead of an array
    print("zdata", zdata.shape, zdata.dtype)
    return zdata

def get_nranges(layer_half_width, stack_half_width):
    nrange = np.arange(-layer_half_width, layer_half_width+1)
    print("nrange", nrange.shape, nrange.dtype, nrange[0], nrange[-1])
    # frange = np.arange(-stack_half_width, stack_half_width+1)
    center = layer_half_width
    mr0 = center - stack_half_width
    mr1 = center + stack_half_width
    fslice = slice(mr0, mr1+1)
    return nrange, fslice

def create_output_volumes(ppm_data, nranges, create_stack):
    xyzs, normals, normals_valid = ppm_data
    nrange, fslice = nranges
    volume = np.zeros((nrange.shape[0],*xyzs.shape[0:2]), dtype=np.uint16)
    print("volume", volume.shape, volume.dtype)
    if create_stack:
        stack_arr = np.zeros(xyzs.shape[0:2], dtype=np.uint16)
        print("stack_arr", stack_arr.shape, stack_arr.dtype)
    else:
        stack_arr = None
        print("no stack_arr")
    return volume, stack_arr

def create_sorted_blocks(ppm_data, step):
    xyzs, normals, normals_valid = ppm_data

    nu = xyzs.shape[1]
    nv = xyzs.shape[0]
    nub = nu // step
    if nub*step < nu:
        nub += 1
    
    nvb = nv // step
    if nvb*step < nv:
        nvb += 1
    
    print("nu,nv,nub,nvb", nu, nv, nub, nvb)
    zsnan = xyzs[...,2].copy()
    zsnan[~normals_valid] = np.nan
    zsp = np.pad(zsnan, ((0,nvb*step-nv), (0,nub*step-nu)), constant_values=np.nan)
    # print("zsp", zsp.shape)
    zsp = zsp.reshape((nvb, step, nub, step))
    zsp = zsp.transpose(0,2,1,3)
    # print("zsp", zsp.shape)
    nvp = np.pad(normals_valid, ((0,nvb*step-nv), (0,nub*step-nu)))
    nvp = nvp.reshape((nvb, step, nub, step))
    # print("nvp", nvp.shape, nvp.dtype)
    nvpa = nvp.any(axis=(1,3))
    # print("nvpa", nvpa.shape, nvpa.dtype)
    zspb = zsp[nvpa]
    # print("zspb", zspb.shape)
    zspa = np.full((nvb, nub), np.nan, dtype=np.float32)
    zspa[nvpa] = np.nanmean(zspb, axis=(1,2))
    # print("zspa", zspa.shape)
    uvs = np.mgrid[:nvb, :nub].transpose(1,2,0).astype(np.float32)
    zuvs = np.concatenate((zspa[...,np.newaxis], uvs), axis=2)
    # print("zuvs", zuvs.shape)
    zuvs = zuvs[nvpa]
    # print("zuvs", zuvs.shape)
    # print(zuvs[0], zuvs[-1])
    sorter = np.argsort(zuvs[:,0])
    zuvs = zuvs[sorter]
    # print(zuvs[0], zuvs[-1])
    print("block count", len(zuvs))
    return zuvs

def process_block(block, step, ppm_data, zdata, nranges, out_volumes, i):
    xyzs, normals, normals_valid = ppm_data
    nrange, fslice = nranges
    volume, stack_arr = out_volumes
    u = int(block[2])*step
    v = int(block[1])*step
    z = block[0]
    # testing
    # if z > 10000.:
    #     break

    u1 = min(u+step, xyzs.shape[1])
    v1 = min(v+step, xyzs.shape[0])
    if i%100 == 0:
        print("block", i, "avg z", z, " u range", u, u1, " v range", v, v1, "        ", end='\r')
    lxyzs = xyzs[v:v1,u:u1,:]
    lnormals = normals[v:v1,u:u1,:]
    lvalid = normals_valid[v:v1,u:u1]
    if not lvalid.any():
        return
    nn = nrange.shape[0]
    luvws = np.mgrid[:nn, v:v1,u:u1].transpose(1,2,3,0)
    lxyzns = lxyzs[np.newaxis,:,:,:]+nrange[:,np.newaxis,np.newaxis,np.newaxis]*lnormals[np.newaxis,:,:,:]
    # print(lxyzns[:,lvalid,:].shape)
    xmin = int(lxyzns[..., 0][:,lvalid].min())
    xmax = int(lxyzns[..., 0][:,lvalid].max()+1)
    ymin = int(lxyzns[..., 1][:,lvalid].min())
    ymax = int(lxyzns[..., 1][:,lvalid].max()+1)
    zmin = int(lxyzns[..., 2][:,lvalid].min())
    zmax = int(lxyzns[..., 2][:,lvalid].max()+1)
    xmin = max(xmin,0)
    ymin = max(ymin,0)
    zmin = max(zmin,0)
    # print("lxyzns", lxyzns.shape, lxyzns.dtype)
    # print(xmin,xmax,ymin,ymax,zmin,zmax)
    data = zdata[zmin:zmax+1,ymin:ymax+1,xmin:xmax+1].astype(np.float32)
    if not data.any():
        return
    xs = np.arange(xmin, xmax+1)
    ys = np.arange(ymin, ymax+1)
    zs = np.arange(zmin, zmax+1)
    # the trilinear interpolator seems slightly faster
    # interp = RegularGridInterpolator((zs,ys,xs), data, bounds_error=False, fill_value=0)
    zyx0 = np.array((zmin,ymin,xmin))
    # xyz0 = np.array((xmin,ymin,zmin))
    zyxs = lxyzns[:,:,:,(2,1,0)]

    # ivals = interp(zyxs)
    # print(zyxs.min(), zyxs.max())
    zyxs[:,lvalid] -= zyx0
    # print(zyxs.min(), zyxs.max())
    ivals = trilinear(zyxs, data)
    ivals[:,~lvalid] = 0
    ivals = np.maximum(ivals, 0)
    ivals = np.minimum(ivals, 65535)
    volume[:,v:v1,u:u1] = ivals.astype(np.uint16)
    if stack_arr is not None:
        stack_arr[v:v1,u:u1] = ivals[fslice].max(axis=0).astype(np.uint16)

def write_volumes(ofl, sfl, volumes):
    volume, stack_arr = volumes

    for i in range(volume.shape[0]):
        fname = "%02d.tif"%i
        print(fname, end='\r')
        tifffile.imwrite(ofl/fname, volume[i,:,:])
    # write stack file last in case there is some problem
    # creating it
    if sfl is not None and stack_arr is not None:
        tifffile.imwrite(sfl.with_suffix(".tif"), stack_arr)
    print()

def create_flattened_layers(params, create_stack):
    ppmfl, zdir, step, maxgb, layer_half_width, stack_half_width = params
    # xyzs, normals, valid_normals
    ppmdata = get_ppm_data(ppmfl)
    if ppmdata is None:
        return
    zdata = get_zarr_data(zdir, maxgb)
    if zdata is None:
        return
    nranges = get_nranges(layer_half_width, stack_half_width)
    output_volumes = create_output_volumes(ppmdata, nranges, create_stack)
    sorted_blocks = create_sorted_blocks(ppmdata, step)
    for i, block in enumerate(sorted_blocks):
        # for testing
        # if i > 100:
        #     break
        process_block(block, step, ppmdata, zdata, nranges, output_volumes, i)
    prev = zdata.store.prev
    print("\nmisses", len(prev.values()), sum([v>1 for v in prev.values()]))
    return output_volumes

def main():
    '''
    odirpath, sfpath = get_output_paths()
    # Do this early to make sure directory can be created

    # ppmfl, zdir, step, maxgb, layer_half_width, stack_half_width
    params = get_params()
    if params is None:
        return
    '''
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Create flattened layers (tifs) given a ppm file and a zarr data store")
    parser.add_argument(
            "ppm_file",
            help="PPM file to use for flattening")
    parser.add_argument(
            "zarr_directory",
            help="Directory containing the scroll data in zarr format")
    parser.add_argument(
            "output_tiff_directory",
            help="Directory where the flattened layers (tiff files) will be written")
    parser.add_argument(
            "output_stack_tiff_file",
            nargs='?',
            help="Optional: name of the output tiff file to contain a stack of the inner layers")
    parser.add_argument(
            "--number_of_layers",
            type=int,
            default=65,
            help="Number of flattened layers to create")
    parser.add_argument(
            "--number_of_stacked_layers",
            type=int,
            default=25,
            help="Number of inner layers to use for the stacked output")
    parser.add_argument(
            "--block_size",
            type=int,
            default=100,
            help="Advanced: size of block to use for subdividing ppm grid")
    parser.add_argument(
            "--zarr_cache_max_size_gb",
            type=int,
            default=8,
            help="Advanced: maximum size in gigabytes of the zarr cache")

    args = parser.parse_args()
    # params: ppmfl, zdir, step, maxgb, layer_half_width, stack_half_width

    params = (
            Path(args.ppm_file),
            Path(args.zarr_directory),
            args.block_size,
            args.zarr_cache_max_size_gb,
            args.number_of_layers // 2,
            args.number_of_stacked_layers // 2)

    odirpath = Path(args.output_tiff_directory)
    sfpath = args.output_stack_tiff_file
    if sfpath is not None:
        sfpath = Path(sfpath)

    flag = create_output_dir(odirpath)
    if not flag:
        return

    create_stack = sfpath is not None

    volumes = create_flattened_layers(params, create_stack)
    write_volumes(odirpath, sfpath, volumes)

if __name__ == '__main__':
    sys.exit(main())
    
