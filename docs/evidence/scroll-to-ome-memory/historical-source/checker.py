import itertools
import json
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, bounded_sizes, check_deadline, fixture_seals, parser, read_json, require, seal, source_seals, write_json
from runtime import environment


class CorrectnessError(Exception):
    pass


def exact(ok, message):
    if not ok:
        raise CorrectnessError(message)


def main():
    cli = parser()
    cli.add_argument('--deadline', required=True, type=float)
    args = cli.parse_args()
    protocol = read_json(HERE / 'protocol.json')
    _, np, tifffile, zarr, identity, before = environment(args, protocol)
    fixture = read_json(args.out / 'fixture.json')
    require(fixture['status'] == 'FIXTURE_SEALED' and fixture['environment'] == identity, 'fixture environment')
    require(fixture['sources'] == before, 'generator source identity')
    require(fixture_seals(args.out, protocol, args.deadline) == fixture['seals'], 'TIFF pre/post seals')
    for z in range(32):
        check_deadline(args.deadline)
        with tifffile.TiffFile(args.out / 'fixture' / ('%02d.tif' % z)) as tif:
            require(len(tif.pages) == 1 and tif.byteorder == '<', 'single little-endian TIFF page')
            page = tif.pages[0]
            require(page.shape == (2048, 2048) and page.dtype == np.dtype('<u2'), 'TIFF shape dtype')
            require(int(page.compression) == 1 and int(page.photometric) == 1 and page.rowsperstrip == 64 and not page.is_tiled, 'TIFF layout')
            require('Predictor' not in page.tags or page.tags['Predictor'].value == 1, 'no TIFF predictor')
            require(len(page.dataoffsets) == 32 and all(v == 262144 for v in page.databytecounts), 'TIFF strips')
            pixels = page.asarray()
            for y in range(0, 2048, 32):
                check_deadline(args.deadline)
                ys = np.arange(y, y + 32, dtype=np.uint32)[:, None]
                xs = np.arange(2048, dtype=np.uint32)[None, :]
                expected_tiff = ((z * 257 + ys * 17 + xs * 31) % 65536).astype('<u2')
                require(np.array_equal(pixels[y:y + 32, :], expected_tiff), 'invalid fixture coordinate oracle')
            del pixels
    bounded_sizes(args.out, protocol)
    binding = {'environment': identity, 'sources': before, 'fixture_post_seals': fixture['seals'], 'fixture_valid': True}
    try:
        exact(sorted(p.name for p in (args.out / 'outputs').iterdir()) == ['a1', 'a2', 'a3', 'a4'], 'exact output cohort')
        arrays = []
        output_seals = {}
        expected_files = {'.zarray'} | {'0/%d/%d' % (y, x) for y in range(64) for x in range(64)}
        for spec in protocol['arms']:
            check_deadline(args.deadline)
            receipt = read_json(args.out / (spec['id'] + '.json'))
            require(receipt['environment'] == identity and receipt['sources'] == before and receipt['returned_is_none'] is True, 'arm environment/source/completion')
            directory = args.out / 'outputs' / spec['id']
            actual_files = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
            exact(actual_files == expected_files, 'exact persisted Zarr file cohort')
            try:
                metadata = read_json(directory / '.zarray')
            except json.JSONDecodeError as error:
                raise CorrectnessError('invalid persisted Zarr metadata') from error
            exact(isinstance(metadata, dict), 'Zarr metadata object')
            exact(metadata.get('order') == 'C' and metadata.get('compressor') is None and metadata.get('filters') is None and metadata.get('zarr_format') == 2, 'Zarr encoding')
            try:
                array = zarr.open_array(str(directory), mode='r')
            except (ValueError, TypeError, KeyError) as error:
                raise CorrectnessError('invalid persisted Zarr array') from error
            exact(array.shape == (32, 2048, 2048) and array.chunks == (32, 32, 32) and array.dtype == np.dtype('uint8'), 'Zarr shape dtype chunks')
            arrays.append(array)
            output_seals[spec['id']] = {}
            for relative in sorted(actual_files):
                check_deadline(args.deadline)
                row = seal(directory / relative)
                exact(relative == '.zarray' or row['bytes'] == 32768, 'exact raw chunk bytes')
                output_seals[spec['id']][relative] = row
        checked = 0
        for z0, y0, x0 in itertools.product(range(0, 32, 32), range(0, 2048, 32), range(0, 2048, 32)):
            check_deadline(args.deadline)
            zz = np.arange(z0, z0 + 32, dtype=np.uint32)[:, None, None]
            yy = np.arange(y0, y0 + 32, dtype=np.uint32)[None, :, None]
            xx = np.arange(x0, x0 + 32, dtype=np.uint32)[None, None, :]
            expected = (((zz * 257 + yy * 17 + xx * 31) % 65536) // 256).astype('uint8')
            first = None
            for array in arrays:
                try:
                    block = array[z0:z0 + 32, y0:y0 + 32, x0:x0 + 32]
                except (ValueError, TypeError, KeyError) as error:
                    raise CorrectnessError('invalid persisted Zarr chunk') from error
                exact(np.array_equal(block, expected), 'every persisted coordinate oracle')
                exact(first is None or np.array_equal(block, first), 'cross-arm exact values')
                if first is None:
                    first = block
            checked += expected.size
        require(checked == 134217728, 'complete coordinate coverage per arm')
    except CorrectnessError as error:
        require(source_seals() == before and fixture_seals(args.out, protocol, args.deadline) == fixture['seals'], 'rejection source/TIFF seals')
        bounded_sizes(args.out, protocol)
        check_deadline(args.deadline)
        write_json(args.out / 'checker.json', {**binding, 'status': 'REJECT_WITNESS_CORRECTNESS', 'failure': str(error)})
        return 1
    require(source_seals() == before and fixture_seals(args.out, protocol, args.deadline) == fixture['seals'], 'final source/TIFF seals')
    sizes = bounded_sizes(args.out, protocol)
    write_json(args.out / 'checker.json', {'status': 'CHECKED_ALL_COORDINATES', 'coordinates_per_arm': checked,
               **binding, 'output_seals': output_seals, 'sizes_before_receipt': sizes})
    bounded_sizes(args.out, protocol)
    check_deadline(args.deadline)


if __name__ == '__main__':
    try:
        code = main()
    except BaseException as error:
        sys.stderr.write(type(error).__name__ + ': ' + str(error) + '\n')
        code = 2
    raise SystemExit(0 if code is None else code)
