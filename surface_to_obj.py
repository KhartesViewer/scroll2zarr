import sys
from pathlib import Path
import argparse
import numpy as np
import tifffile
import scipy
import cv2
from scipy.spatial import Delaunay

from ppm import Ppm

# Modifies flags array; points that are not to be
# used are set to False.
# Only the first 3 elements (x,y,z) of xyzs are used
def adaptiveDecimate1d(xyzs, flags, axis, min_ratio, min_length, level):
    trxs = xyzs.transpose(axis, 1-axis, 2)
    trfs = flags.transpose(axis, 1-axis)
    n2 = 2**level
    n2p1 = n2*2
    nf = len(trfs)
    nfw = trfs.shape[1]
    nint = (nf-1) // n2p1
    if nint == 0:
        return
    # nte is length that is truncated to nint*n2p1, and
    # then extended to next multiple of n2p1
    nte = n2p1*(nint+1)
    tefs = np.zeros((nte,nfw), dtype=trfs.dtype)
    tefs[:nf,:] = trfs
    # print(trfs)
    # print(tefs)
    folded = tefs.reshape(nint+1,n2p1,nfw)
    # print(folded)
    folded = folded[:,1:,:]
    folded = np.delete(folded, n2-1, axis=1)
    # print(folded)
    use = ~folded.any(axis=1)[:-1,:]
    # print(use)
    # print(trunc)
    texs = np.zeros((nte, nfw, 3), dtype=np.float64)
    texs[:nf,:,:] = trxs[:,:,:3]
    # xyzs at the ends of intervals
    endxs = texs[::n2p1,:,:]
    # print()
    # print()
    # print(endxs)
    # diff between endpoints
    delxs = np.diff(endxs, axis=0)
    # 3d distance between endpoints
    dxs = np.sqrt((delxs*delxs).sum(axis=2))
    # print()
    # print(delxs)
    rxs = np.roll(endxs, -1, axis=0)
    # avg of endpoints
    avgxs = .5*(rxs[:-1]+endxs[:-1])
    # print()
    # print(avgxs)
    # xyz of points in centers of intervals
    cxs = texs[n2::n2p1,:,:][:-1,:,:]
    delca = avgxs-cxs
    # 3d distance between endpoint average and point at center of interval
    dca = np.sqrt((delca*delca).sum(axis=2))
    # print()
    # print(dxs)
    # print(dca)
    ratio = dca/(dxs+.000001)
    
    # print(ratio)
    cfs = trfs[n2::n2p1,:][:use.shape[0],:]
    # print(cfs)
    cfs[use] &= ~((ratio < min_ratio)[use] | (dca < min_length)[use])

# Returns decimated vcpsd points as a (n,m) array where
# n is the number of surviving points, and m is the number
# of elements per point (usually 6: x,y,z,u,v,flag)
def adaptiveDecimate2d(vcpsd, flags, min_ratio, min_length):
    for i in range(12):
        adaptiveDecimate1d(vcpsd, flags, 1, min_ratio, min_length, i)
    for i in range(12):
        adaptiveDecimate1d(vcpsd, flags, 0, min_ratio, min_length, i)
    return vcpsd[flags]

def simpleDecimateTest():
    testshape = (7,2,3)
    
    testflags = np.full(testshape[:2], True)
    # testflags[1,0] = False
    # testflags[2,0] = False
    # testflags[3,0] = False
    
    testxyzs = np.zeros(testshape, dtype=np.float64)
    testxyzs[:,0,2] = np.arange(0., testshape[0])
    testxyzs[:,1,2] = np.arange(0., testshape[0])
    testxyzs[1,0,0] = 2.
    testxyzs[2,0,0] = 3.
    testxyzs[4,0,0] = 5.
    # testxyzs[1,0] = [0., 0., 2.]
    # testxyzs[2,0] = [0., 0., 3.]
    # testxyzs[4,0] = [0., 0., 5.]
    # print(testflags)
    
    decimate(testxyzs, testflags, 0, .2, 0)
    print(testflags)
    decimate(testxyzs, testflags, 0, .2, 1)
    print(testflags)
    exit()

# Note that if smoothing > 0, the input xyzuv grid will be smoothed
def gridDecimate(xyzuv, zstep, zsmoothing):
    d0, d1 = pointSpacing(xyzuv)
    print("d0, d1", d0, d1)
    psr = d0 / d1
    dec0 = 1
    dec1 = 1
    if zstep > 0:
        dec0 = int(zstep+.5)
        dec1 = int(zstep*psr+.5)
    print("dec0,dec1", dec0, dec1)
    
    if zsmoothing > 0:
        sigmas = (zsmoothing, zsmoothing*psr)
        for i in range(3):
            # xuzuv[:,:,i] = scipy.ndimage.gaussian_filter(xuzuv[:,:,i], sigmas, mode='nearest')
            # The OpenCV version of Gaussian blur is faster:
            xyzuv[:,:,i] = cv2.GaussianBlur(xyzuv[:,:,i], (0,0), sigmas[0], sigmas[1], cv2.BORDER_ISOLATED)

    return xyzuv[::dec0,::dec1,:]

def readIndexedPpm(ppm_path):
    ppm = Ppm.loadPpm(ppm_path)
    if not ppm.valid:
        print("Error:", ppm.error)
        return None
    ppm.loadData()
    xyzs = ppm.ijks
    flags = (ppm.normals != 0).any(axis=2)
    print("xyzs", xyzs.shape, xyzs.dtype)
    indexed_vcps = addIndexToXyzs(xyzs)
    indexed_vcps[:,:,5] = flags
    return indexed_vcps

def readIndexedVcps(vcps_path):
    vcps = readVcps(vcps_path)
    if vcps is None:
        return None
    return addIndexToXyzs(vcps)
    
def readVcps(vcps_path):
    try:
        vcps_fd = open(vcps_path, "rb")
    except Exception as err:
        print("Error in readVcps", err)
        return None
    vcps_header = {}
    for line in vcps_fd:
        line = line.decode('utf-8')
        line = line.strip()
        # print("line", line)
        if line == '<>':
            break
        words = line.split()
        if len(words) != 2 or not words[0].endswith(':'):
            print("Unrecognized line", line)
            continue
        key = words[0][:-1]
        value = words[1]
        vcps_header[key] = value
    
    # print(vcps_header)
    
    if 'width' not in vcps_header:
        print("vcps header does not specify width")
        return None
    
    if 'height' not in vcps_header:
        print("vcps header does not specify height")
        return None
    vcps_width = int(vcps_header['width'])
    vcps_height = int(vcps_header['height'])
    expected = {
            'dim': '3',
            'ordered': 'true',
            'type': 'double',
            'version': '1'
            }
    for key in expected.keys():
        if key not in vcps_header:
            print("vcps header is missing", key)
            return None
        if vcps_header[key] != expected[key]:
            print("vcps header key", key, "contains unexpected value", vcps_header[key])
            return None
    
    # print(vcps_width, vcps_height, 3*vcps_width*vcps_height)
    print(vcps_width, vcps_height)
    
    vcps_raw = np.fromfile(vcps_fd, dtype=np.float64)
    # print("shape", vcps_raw.shape)
    if vcps_raw.shape[0] != vcps_width*vcps_height*3:
        print("Expected vcps file to contain",vcps_width*vcps_height*3,"points, contains",vcps_raw.shape[0])
        return None
    
    vcps = vcps_raw.reshape((vcps_height, vcps_width, 3))
    print("shape", vcps.shape)
    return vcps

def addIndexToXyzs(vcps):
    # print(vcps[0][0], vcps[1][0])
    indices = np.indices((vcps.shape[:2]), dtype=np.float64)
    indexed_vcps = np.zeros((vcps.shape[0], vcps.shape[1], 6), dtype=np.float64)
    indexed_vcps[:,:,:3] = vcps
    indexed_vcps[:,:,3:5] = indices.transpose(1,2,0)[:,:,::-1]
    indexed_vcps[:,:,5] = 1
    print("iv", indexed_vcps.shape)
    return indexed_vcps

# returns average xyz distance between points in the
# two directions (d0 is in the z direction, d1 is in the 
# winding direction)
def pointSpacing(indexed_vcps):
    vcps = indexed_vcps[:,:,:3]
    flags = indexed_vcps[:,:,5]
    f0 = np.logical_and(flags[:-1,:], np.roll(flags,-1,axis=0)[:-1,:])
    d0vcps = np.diff(vcps, axis=0)
    d0vcps = np.sqrt((d0vcps*d0vcps).sum(axis=2))
    f1 = np.logical_and(flags[:,:-1], np.roll(flags,-1,axis=1)[:,:-1])
    d1vcps = np.diff(vcps, axis=1)
    d1vcps = np.sqrt((d1vcps*d1vcps).sum(axis=2))
    d0avg = np.average(d0vcps[f0])
    d1avg = np.average(d1vcps[f1])
    # print("avg", np.average(d0vcps), np.average(d1vcps))
    return d0avg, d1avg

class TrglList:
    def __init__(self):
        self.trgls = None

    @staticmethod
    def outsidePoints(uvpts, uvstep):
        uvmin = uvpts.min(axis=0)
        uvmax = uvpts.max(axis=0)
        minid = uvmin/uvstep - 5
        maxid = uvmax/uvstep + 5
        id0 = np.floor(minid).astype(np.int32)
        id1 = np.ceil(maxid).astype(np.int32)
        idn = id1-id0
        # We want to find all the points outside of the obj surface.
        # To do this, create an array, and set all the cells of
        # the array to 255.  Then set all cells that contain a
        # trgl vertex to 0.
        # Look for connected components, and take the connected
        # component that extends to 0,0; that is the outer region.
        # Note that connected components are created from non-zero
        # components, so need to make sure the points in the area 
        # we are interested in is non-zero.
        arr = np.full(idn, 255, dtype=np.uint8)
        istps = np.floor(uvpts/uvstep - id0).astype(np.int32)
        # print("outsidePoints arr", id0, idn, istps, arr.shape)
        # print("outsidePoints arr", id0, idn, arr.shape)
        arr[istps[:,0], istps[:,1]] = 0
        # cv2.imwrite("test.png", arr)
        ccoutput = cv2.connectedComponentsWithStats(arr, 4, cv2.CV_32S)
        (nlabels, labels, stats, centroids) = ccoutput
        label0 = labels[0,0]
        # print("nlabels", nlabels, "label0", label0)
        # print("stats", stats[label0])
        arr2 = np.full(idn, 255, dtype=np.uint8)
        # points that are outside are 0, points not in the outside are 255
        arr2[labels == label0] = 0
        # cv2.imwrite("test.png", arr2)
        kernel = np.ones((3,3), np.uint8)
        dilo = cv2.dilate(arr2, kernel, iterations=2)
        dili = cv2.dilate(arr2, kernel, iterations=1)
        diff = dilo-dili
        # cv2.imwrite("test.png", diff)
        pts = np.argwhere(diff)
        # print("pts", pts.shape, pts[0:5])
        pts = (pts+id0)*uvstep
        # print("pts", pts.shape, pts[0:5])
        return pts
    
    @staticmethod
    def delaunayInternal(iuvpts, uvstep=0., nudge_uv=False):
        uvpts = iuvpts.copy()
        if nudge_uv:
            uvpts[:,0] += .00001*uvpts[:,1]
            uvpts[:,1] += .00001*uvpts[:,0]
        print("triangulating")
        if uvstep > 0.:
            outside_pts = TrglList.outsidePoints(uvpts, uvstep)
            print("input points", len(uvpts))
            print("outside points", len(outside_pts))
            all_pts = np.concatenate((uvpts, outside_pts), axis=0)
        else:
            all_pts = uvpts
        all_trgls = None
        try:
            all_trgls = Delaunay(all_pts).simplices
        except Exception as err:
            err = str(err).splitlines([0])
            print("fromEmbayedDelaunay error: %s"%err)
        new_trgls = None
        if all_trgls is not None:
            new_trgls = all_trgls[(all_trgls < len(uvpts)).all(axis=1), :]
            print("all trgls", len(all_trgls))
            print("new trgls", len(new_trgls))
        tlist = TrglList()
        tlist.trgls = new_trgls
        return tlist

    @staticmethod
    def fromEmbayedDelaunay(uvpts, uvstep, nudge_uv=False):
        return TrglList.delaunayInternal(uvpts, uvstep, nudge_uv)

    @staticmethod
    def fromDelaunay(uvpts):
        return TrglList.delaunayInternal(uvpts, 0., False)

    # returns 3*n array, where each triplet corresponds
    # to a trgl, and each member is the id of the trgl's 
    # neighbor on that side
    def neighbors(self):
        trgls = self.trgls
        if trgls is None:
            return None

        index = np.indices((len(trgls),1))[0]
        ones = np.ones((len(trgls),1), dtype=np.int32)
        zeros = np.zeros((len(trgls),1), dtype=np.int32)
        twos = 2*ones
        
        e01 = np.concatenate((trgls[:, (0,1)], index, twos, ones), axis=1)
        e12 = np.concatenate((trgls[:, (1,2)], index, zeros, ones), axis=1)
        e20 = np.concatenate((trgls[:, (2,0)], index, ones, ones), axis=1)
        
        edges = np.concatenate((e01,e12,e20), axis=0)
        rev = (edges[:,0] > edges[:,1])
        edges[rev,0:2] = edges[rev,1::-1]
        edges[rev,4] = -1
        edges = edges[edges[:,4].argsort()]
        edges = edges[edges[:,1].argsort(kind='mergesort')]
        edges = edges[edges[:,0].argsort(kind='mergesort')]
        
        ediff = np.diff(edges, axis=0)
        duprows = np.where(((ediff[:,0]==0) & (ediff[:,1]==0)))[0]
        duprows2 = np.sort(np.append(duprows, duprows+1))
        bdup = np.zeros((len(edges)), dtype=np.bool_)
        bdup[duprows2] = True
        
        neighbors = np.full((len(trgls), 3), -1, dtype=np.int32)
        
        eplus = edges[duprows+1,:4]
        eminus = edges[duprows,:4]
        # print(eplus)
        # print(eminus)
        neighbors[eplus[:,2],eplus[:,3]] = eminus[:,2]
        neighbors[eminus[:,2],eminus[:,3]] = eplus[:,2]
        return neighbors

    # returns indexes of those trgls that have pt_index as
    # a vertex
    def trglsAroundPoint(self, pt_index):
        trgls = self.trgls
        if trgls is None:
            return None
        bvec = (trgls == pt_index).any(axis=1)
        tindexes = np.where(bvec)[0]
        # return tindexes.tolist()
        return tindexes

    # output: a number (nc, number of compenents) and
    # a 1D array.  The 1d array is the same length as
    # the trgl array, and each element of the array is
    # the id of the connected component that the corresponding
    # trgl belongs to.
    # Note that "connected" is based on trgl-trgl connections;
    # two trgls sharing the same vertex are not necessarily
    # connected (TODO: is that the right approach?)
    def connectedComponentLabels(self):
        trgls = self.trgls
        if trgls is None:
            return 0, None
        neighbors = self.neighbors()
        nt = trgls.shape[0]
        # print("t n", trgls.shape, neighbors.shape)
        # each triangle also has itself as a neighbor
        # (this is so that triangles with no neighbors will still
        # show up in the connectivity graph)
        neighbors = np.append(neighbors, np.ogrid[:nt,:0][0], axis=1)
        tindex = np.ogrid[:4*nt]//4
        nindex = neighbors.flatten()
        is_valid = (nindex > -1)
        tindex = tindex[is_valid]
        nindex = nindex[is_valid]
        ones = np.full(nindex.shape[0], 1)
        connections = scipy.sparse.csr_array((ones, (tindex, nindex)), shape=(nt,nt))
        # print("connections")
        # print(connections)
        nc, labels = scipy.sparse.csgraph.connected_components(connections, directed=False)
        return nc, labels

    # return list of tuples.  Each tuple represents a single
    # connected component.  Each tuple consists of two
    # items: an array, length n-old-points, having the
    # new-point id for each old point id (or -1 if no new point)
    # and a TrglList.
    def connectedComponents(self):
        nc, labels = self.connectedComponentLabels()
        if nc == 0:
            return None
        # print("connected components", nc)
        components = []
        for i in range(nc):
            tl = TrglList()
            tl.trgls = self.trgls[i==labels]
            pts = tl.renumberTrgls()
            # print("  ", len(pts), len(tl.trgls))
            components.append((pts, tl))
        return components

    # renumbers trgls in trgl list so that all point ids consist
    # of points that are actually used by one or more trgl.
    # Returns array of length n, where n is the number
    # of old points; each item contains the id of the corresponding
    # point in the new trgl list.  If the item in the old list
    # is set to -1, then that point is not used in by any trgl
    # in the list.
    def renumberTrgls(self):
        trgls = self.trgls
        if trgls is None:
            return None
        if len(trgls) == 0:
            return None
        onpts = trgls.max() + 1
        fpts = np.zeros((onpts), dtype=np.bool_)
        fpts[trgls.flatten()] = True
        nnpts = fpts.sum()
        opts = np.full((onpts), -1, dtype=np.int32)
        opts[fpts] = np.arange(nnpts)
        self.trgls = opts[trgls]
        return opts

    @staticmethod
    def renumberUsingPts(pts, xyz, uv):
        npts = pts.shape[0]
        lxyz = xyz[:npts][pts > -1]
        luv = uv[:npts][pts > -1]
        return lxyz, luv

    # pts is a n*2 or n*3 array, where n is the number
    # of points, and each point consists of either 2 or 3
    # values (uv or xyz, say).  The mins and maxs must
    # have the same size (2 or 3) as pts.
    # Returns a new TrglList with all the trgls that are
    # at least partially in the min-max range
    def window(self, pts, mins, maxs):
        if self.trgls is None:
            return None
        f = ((pts >= mins) & (pts <= maxs)).all(axis=1)
        tl = TrglList()
        tl.trgls = self.trgls[f[self.trgls].any(axis=1)]
        return tl

    def saveAsObjFile(self, xyz, uv, obj_name):
        obj_fd = open(obj_name, "w")
        print("saving", obj_name.name, len(xyz), len(self.trgls))
        for pt in xyz:
            print("v", pt[0], pt[1], pt[2], file=obj_fd)
        for pt in uv:
            print("vt", pt[0], pt[1], file=obj_fd)
        for trgl in self.trgls:
            print("f", trgl[0]+1, trgl[1]+1, trgl[2]+1, file=obj_fd)

    def saveAsObjFiles(self, xyz, uv, obj_name):
        self.saveAsObjFile(xyz, uv, obj_name)
        components = self.connectedComponents()
        if components is None:
            print("saveAsObjFiles: problem with connected components")
            return
        stem = obj_name.stem
        suff = obj_name.suffix
        op = obj_name.parent
        # print("saof nxyz", len(xyz))
        if len(components) > 1:
            for i,c in enumerate(components):
                pts, tl = c
                lxyz, luv = self.renumberUsingPts(pts, xyz, uv)
                oname = (obj_name.parent / ("%s%03d"%(stem,i+1))).with_suffix(suff)
                tl.saveAsObjFile(lxyz, luv, oname)


    @staticmethod
    def loadFromObjFile(obj_file):
        print("loading obj file", obj_file)
        pname = Path(obj_file)
        try:
            fd = pname.open("r")
        except:
            print("Could not load obj file", pname)
            return None
        
        # v list
        vrtl = []
        # tv list
        tvrtl = []
        # trgl v's
        vtrgl = []
        # trgl tv's
        tvtrgl = []
        
        created = ""
        frag_name = ""
        for line in fd:
            line = line.strip()
            words = line.split()
            if words == []: # prevent crash on empty line
                continue
            if words[0][0] == '#':
                if len(words) > 2: 
                    if words[1] == "Created:":
                        created = words[2]
                    if words[1] == "Name:":
                        frag_name = words[2]
            elif words[0] == 'v':
                # len is 7 if the vrt has color attached
                # (color is ignored)
                if len(words) == 4 or len(words) == 7:
                    vrtl.append([float(w) for w in words[1:4]])
            elif words[0] == 'vt':
                if len(words) == 3:
                    tvrtl.append([float(w) for w in words[1:]])
            elif words[0] == 'f':
                if len(words) == 4:
                    # old code:
                    # # implicit assumption that v == vt
                    # trgl.append([int(w.split('/')[0])-1 for w in words[1:]])

                    # in the 'f' lines, v and vt may be different.
                    # save both for later processing
                    vs = []
                    tvs = []
                    for word in words[1:]:
                        ws = word.split('/')
                        v = int(ws[0])-1
                        if len(ws) > 1:
                            tv = int(ws[1])-1
                        else:
                            tv = v
                        vs.append(v)
                        tvs.append(tv)
                    vtrgl.append(vs)
                    tvtrgl.append(tvs)


        print("obj reader", len(vrtl), len(tvrtl), len(vtrgl))

        if len(vtrgl) > 0:
            vtrg = np.array(vtrgl, dtype=np.int32)
            tvtrg = np.array(tvtrgl, dtype=np.int32)
        else:
            vtrg = np.zeros((0,3), dtype=np.int32)
            tvtrg = np.zeros((0,3), dtype=np.int32)
        if len(vrtl) > 0:
            vrt = np.array(vrtl, dtype=np.float32)
        else:
            vrt = np.zeros((0,3), dtype=np.float32)
        if len(tvrtl) > 0:
            tvrt = np.array(tvrtl, dtype=np.float32)
        else:
            tvrt = np.zeros((0,2), dtype=np.float32)
        otvrt = np.zeros((len(vrt), 2), dtype=np.float32)
        # otvrt will contain uv values that are listed
        # in the same order as the xyz values in vrt.
        # print(vtrg[:3])
        # print(tvtrg[:3])
        otvrt[vtrg.flatten()] = tvrt[tvtrg.flatten()]
        # print("otvrt")
        # print(otvrt[:5])
        tl = TrglList()
        tl.trgls = vtrg
        xyzs = vrt
        uvs = otvrt
        return tl, xyzs, uvs

def sphericalDecimateTest(idir):
    sz = 129
    szf = (sz-1)//2
    vcps = np.zeros((sz,sz,3), dtype=np.float64)
    indexed_vcps = addIndexToXyzs(vcps)
    indexed_vcps[:,:,0] = indexed_vcps[:,:,3]/szf
    indexed_vcps[:,:,1] = indexed_vcps[:,:,4]/szf
    vxy = (indexed_vcps[:,:,:2] - 1.)
    indexed_vcps[:,:,2] = np.sqrt(2-(vxy*vxy).sum(axis=2))
    
    vcpsd = indexed_vcps
    ratio = .003
    length = 4*ratio/szf
    flags = np.full(vcpsd.shape[:2], True)
    flags = indexed_vcps[:,:,2] > 1
    for i in range(12):
        adaptiveDecimate1d(vcpsd, flags, 0, ratio, length, i)
        adaptiveDecimate1d(vcpsd, flags, 1, ratio, length, i)
    
    vcpsd = vcpsd[flags]
    
    obj_name = idir/"test.obj"
    saveAsObjFiles(vcpsd, obj_name)

def parseSlices(istr, dim):
    sstrs = istr.split(",")
    if len(sstrs) != dim:
        print("Could not parse range argument '%s'; expected %d comma-separated ranges"%(istr,dim))
        return [None]*dim
    slices = []
    for sstr in sstrs:
        if sstr == "":
            slices.append(None)
            continue
        parts = sstr.split(':')
        if len(parts) == 1:
            slices.append(slice(int(parts[0])))
        else:
            iparts = [None if p=="" else float(p) for p in parts]
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

def windowMinMax(istr, pts):
    owin = np.array([pts.min(axis=0)-.001, pts.max(axis=0)+.001])
    # print("owin", owin)
    wmin = owin[0,:]
    wmax = owin[1,:]
    if istr is None:
        return owin

    dpts = pts.shape[1]

    slices = parseSlices(istr, dpts)
    for i,s in enumerate(slices):
        if s is None:
            continue
        if not slice_step_is_1(s):
            print("slice step must be 1", istr)
        wmin[i] = slice_start(s, wmin[i])
        wmax[i] = slice_stop(s, wmax[i])
    # print("owin2", owin)
    return owin
    
def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Create obj file from ppm, vcps, or obj file; apply decimation if desired")
    parser.add_argument(
            "input_surface_file",
            help="obj, ppm, or vcps file")
    parser.add_argument(
            "output_obj_file",
            help="Name of output obj file")
    parser.add_argument(
            "--zstep",
            type=int,
            default=24,
            help="z decimation step size (xy step size will be proportional to this); 0 means no decimation")
    parser.add_argument(
            "--zsmooth",
            type=float,
            default=6.,
            help="z decimation smoothing length (xy smoothing length will be proportional to this); 0 means no smoothing")
    parser.add_argument(
            "--xyzwindow",
            help="Output only that part of the surface that lies within the given xyz range.  Example (in xyz coordinates): 2500:3000,1500:2000,5000:5500")
    parser.add_argument(
            "--uvwindow",
            help="Output only that part of the surface that lies within the given uv range.  Example (in uv coordinates): 2000:4000,:")
    args = parser.parse_args()

    isurf = Path(args.input_surface_file)
    oobj = Path(args.output_obj_file)

    zstep = args.zstep
    zsmooth = args.zsmooth

    suffix = isurf.suffix[1:]
    print("isurf", isurf)
    print("suffix", suffix)
    if suffix in ["ppm", "vcps"]:
        if suffix == "ppm":
            indexed_vcps = readIndexedPpm(isurf)
            if indexed_vcps is None:
                return()
            # ppm grids cannot be smoothed, due
            # to irregular boundary
            zsmooth = 0
        elif suffix == "vcps":
            indexed_vcps = readIndexedVcps(isurf)
            if indexed_vcps is None:
                return()

        # print("indexed_vcps shape", indexed_vcps.shape)
        vcpsd = gridDecimate(indexed_vcps, zstep, zsmooth)
        vcpsd = vcpsd.reshape(-1,6)
        # Only keep valid points (ignore invalid ppm points)
        vcpsd = vcpsd[vcpsd[:,5]!=0]
        xyzpts = vcpsd[:,:3]
        uvpts = vcpsd[:,3:5]

        d0, d1 = pointSpacing(indexed_vcps)
        print("d0, d1", d0, d1)
        psr = d0 / d1
        deratioed_uvpts = uvpts / [1, psr]
        # use original uvpts for windowing, use de-ratioed uvpts
        # for delaunay and adaptive decimation
        baysize = 2*zstep 
        tlist = TrglList.fromEmbayedDelaunay(deratioed_uvpts, baysize, True)
    elif suffix == "obj":
        result = TrglList.loadFromObjFile(isurf)
        if result is None:
            return()
        tlist, xyzpts, uvpts = result

    uvwin = windowMinMax(args.uvwindow, uvpts)
    if uvwin is not None:
        tlist = tlist.window(uvpts, uvwin[0], uvwin[1])
    xyzwin = windowMinMax(args.xyzwindow, xyzpts)
    if xyzwin is not None:
        tlist = tlist.window(xyzpts, xyzwin[0], xyzwin[1])
    if tlist is None or len(tlist.trgls) == 0:
        print("No triangles to export")
        return
    pts = tlist.renumberTrgls()
    lxyz, luv = TrglList.renumberUsingPts(pts, xyzpts, uvpts)

    tlist.saveAsObjFiles(lxyz, luv, oobj)

if __name__ == '__main__':
    sys.exit(main())
