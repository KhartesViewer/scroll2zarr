import argparse
import shutil
from pathlib import Path
from PIL import Image
# import numpy as np

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="read TIFF files and write them as uncompressed TIFF files")
parser.add_argument(
        "input_tiff_dir",
        help="Directory containing the TIFF files to be decompressed")
parser.add_argument(
        "output_tiff_dir",
        help="Directory where the decompressed TIFF files will be written")
parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory, if it already exists")

args = parser.parse_args()
idir = Path(args.input_tiff_dir)
odir = Path(args.output_tiff_dir)
overwrite = args.overwrite

if odir.exists():
    if overwrite:
        print("Removing existing",odir)
        shutil.rmtree(odir)
    else:
        print("Error: Directory",odir,"already exists")
        exit()

try:
    odir.mkdir()
except Exception as e:
    print("Error while creating",odir,":",e)
    exit()

if not idir.exists():
    print("Input directory",idir,"does not exist")

# idir = r"G:\Vesuvius\Scroll1.volpkg\volumes\20230205180739"
# idir = r"F:\Vesuvius\Scroll1.volpkg\volumes_masked\20230205180739"
# odir = r"F:\Vesuvius\volumes_masked_uc"

files = sorted(idir.glob('*.tif'))
for fl in files:
    oname = fl.name
    print("reading",oname,end='\r')
    # Can't use cv2 because OpenCV automatically compresses
    # TIFF files on writing
    iarr = Image.open(str(fl))
    ofl = odir / oname
    print("writing",oname,end='\r')
    iarr.save(str(ofl), compression=None)
