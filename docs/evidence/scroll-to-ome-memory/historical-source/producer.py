import argparse
import ast
import gc
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
import traceback

CASES = (
    ('partial-conversion', 3, 'uint16', 1),
    ('full-conversion', 2, 'uint16', 1),
    ('partial-preserve-u16', 3, 'uint16', 0),
    ('full-preserve-u16', 2, 'uint16', 0),
    ('partial-u8-control', 3, 'uint8', 1),
    ('full-u8-control', 2, 'uint8', 1),
)
U16 = [[0, 256], [65280, 65535]]
U8 = [[0, 1], [255, 255]]
OLD = b'                tzarr[zs:ze,ys:ye,:] = buf[:ze-zs,:ye-ys,:] // divisor\n'
NEW = b'                tzarr[zs:ze,ys:ye,:] = buf[:ze-zs,:ye-ys,:]\n'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def seal(path):
    safe_file(Path(path))
    raw = Path(path).read_bytes()
    return {'path': str(Path(path).resolve()), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


class DependencyError(Exception):
    pass


def setup(ok, message):
    if not ok:
        raise DependencyError(message)


def check_environment(args, protocol):
    setup(protocol['expected']['zarr_version'] == '2.18.7', 'frozen Zarr 2.18.7 required')
    setup(args.environment_lock is not None, 'NOT_READY_ENVIRONMENT_LOCK')
    lock = json.loads(args.environment_lock.read_text())
    setup(lock['schema'] == 'scroll2zarr-root-environment-v1', 'environment lock schema')
    setup(str(Path(sys.executable).resolve()) == str(args.interpreter.resolve())
          == lock['interpreter'], 'explicit interpreter binding')
    setup(str(args.isolated_env.resolve()) == lock['isolated_env']
          and args.isolated_env.is_dir(), 'explicit isolated env binding')
    setup(str(Path(sys.prefix).resolve()) == lock['isolated_env'], 'active environment prefix')
    setup(list(sys.version_info[:3]) == lock['python'] and sys.version_info[:2] != (3, 14),
          'locked Python required; no 3.14 fallback')
    import numpy
    import tifffile
    import numcodecs
    import zarr
    deps = {}
    for name, mod in (('numpy', numpy), ('tifffile', tifffile),
                      ('numcodecs', numcodecs), ('zarr', zarr)):
        origin = str(Path(mod.__file__).resolve())
        setup(origin.startswith(lock['isolated_env'] + '/'), name + ' origin outside env')
        deps[name] = {'version': str(mod.__version__), 'origin': origin}
        setup(deps[name] == lock['dependencies'][name], name + ' root lock mismatch')
    setup(deps['zarr']['version'] == '2.18.7', 'frozen Zarr 2.18.7 required')
    nc = str(numcodecs.__version__)
    setup(re.fullmatch(r'0\.(10|11|12|13|14|15)\.[0-9]+', nc) is not None
          and nc not in ('0.14.0', '0.14.1'), 'unsupported numcodecs release')
    numpy_version = re.fullmatch(r'([0-9]+)\.([0-9]+)\.[0-9]+', str(numpy.__version__))
    setup(numpy_version is not None and tuple(map(int, numpy_version.groups())) >= (1, 24),
          'unsupported numpy release')
    setup(all(hasattr(zarr, name) for name in ('NestedDirectoryStore', 'open', 'open_array')),
          'genuine Zarr 2 API required')
    setup(callable(tifffile.imread) and callable(tifffile.imwrite), 'genuine TIFF API required')
    return numpy, tifffile, zarr, deps


def safe_file(path):
    path = Path(path)
    require(path.is_absolute() and str(path) == str(path.resolve()), 'canonical file path')
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink forbidden')
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'regular unshared file required')
    require(0 < info.st_size <= 1048576, 'tiny file bound')
    return info.st_dev, info.st_ino


def inventory(directory):
    require(directory.is_dir() and str(directory) == str(directory.resolve()), 'canonical directory')
    require(not any(p.is_symlink() for p in (directory, *directory.parents)), 'symlink forbidden')
    rows = []
    seen = set()
    for path in sorted(directory.rglob('*')):
        require(not path.is_symlink(), 'symlink forbidden')
        if path.is_dir():
            continue
        identity = safe_file(path)
        require(identity not in seen, 'file alias')
        seen.add(identity)
        rows.append({'relative': path.relative_to(directory).as_posix(), **seal(path)})
        require(len(rows) <= 128, 'tiny inventory bound')
    return rows


def extract(path, binding, np, tifffile, zarr):
    raw = path.read_bytes()
    require(len(raw) == binding['bytes'] and hashlib.sha256(raw).hexdigest() == binding['sha256'],
            'source identity')
    tree = ast.parse(raw, filename=str(path))
    lines = raw.splitlines(keepends=True)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'tifs2zarr']
    require(len(selected) == 1, 'exact source function definition required')
    node = selected[0]
    text = b''.join(lines[node.lineno - 1:node.end_lineno])
    actual = {'name': node.name, 'start': node.lineno, 'end': node.end_lineno,
              'source_sha256': hashlib.sha256(text).hexdigest(),
              'ast_sha256': hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()}
    require(actual == binding['functions'][0], 'source span and AST seals')
    namespace = {'np': np, 'tifffile': tifffile, 're': re, 'zarr': zarr}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['tifs2zarr']



def main():
    parser = argparse.ArgumentParser(description='Produce the frozen twelve-case extracted-function capture.')
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--fixed', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--interpreter', type=Path, required=True,
                        help='explicit future interpreter path supplied by root; not an approval')
    parser.add_argument('--isolated-env', type=Path, required=True,
                        help='explicit future isolated environment root supplied by root')
    parser.add_argument('--environment-lock', type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    require(not out.exists(), 'fresh output required')
    out.mkdir(parents=True)
    report = {'status': 'INCOMPLETE', 'boundary': 'GENUINE_EXTRACTED_FUNCTION_PERSISTENCE_UNVERIFIED',
              'capture_root': str(out), 'cases': []}
    code = 2
    try:
        protocol = json.loads(args.protocol.read_text())
        report['protocol'] = seal(args.protocol)
        require(protocol['schema'] == 'scroll2zarr-genuine-extracted-run-v2', 'protocol schema')
        sources = {'baseline': args.original.resolve(), 'fixed': args.fixed.resolve()}
        before = {arm: seal(p) for arm, p in sources.items()}
        for arm in sources:
            require(before[arm]['sha256'] == protocol['sources'][arm]['sha256']
                    and before[arm]['bytes'] == protocol['sources'][arm]['bytes'],
                    arm + ' source binding')
        original = args.original.read_bytes()
        fixed = args.fixed.read_bytes()
        require(original.count(OLD) == 1 and original.replace(OLD, NEW, 1) == fixed,
                'only accepted one-line change')
        setup(args.environment_lock is not None, 'NOT_READY_ENVIRONMENT_LOCK')
        descriptors = {'protocol': args.protocol.resolve(),
                       'environment_lock': args.environment_lock.resolve()}
        report['descriptors_before'] = {name: seal(path) for name, path in descriptors.items()}
        require(report['descriptors_before']['protocol'] == report['protocol'], 'protocol changed')
        try:
            np, tifffile, zarr, dependencies = check_environment(args, protocol)
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise DependencyError('environment descriptor invalid: ' + str(error)) from error
        report['dependencies'] = dependencies
        report['interpreter'] = {'argv': str(args.interpreter), 'executable': sys.executable,
                                 'version': sys.version, 'isolated_env': str(args.isolated_env.resolve())}
        for arm, source in sources.items():
            function = extract(source, protocol['sources'][arm], np, tifffile, zarr)
            for label, planes, dtype, obytes in CASES:
                case_dir = out / arm / label
                inputs = case_dir / 'tiffs'
                inputs.mkdir(parents=True)
                literal = U16 if dtype == 'uint16' else U8
                fixtures = []
                for index in range(planes):
                    path = inputs / ('%05d.tif' % index)
                    tifffile.imwrite(path, np.array(literal, dtype=dtype),
                                     compression=None, photometric='minisblack')
                    read = tifffile.imread(path)
                    require(read.dtype == np.dtype(dtype) and read.tolist() == literal,
                            'real TIFF literal roundtrip')
                    fixtures.append(seal(path))
                input_inventory = inventory(inputs)
                require([dict((k, x[k]) for k in ('path', 'bytes', 'sha256')) for x in input_inventory] == fixtures, 'complete TIFF inventory')
                store_dir = case_dir / 'output.zarr'
                try:
                    returned = function(inputs, store_dir, 2, obytes=obytes, slices=None, maxgb=None)
                except (TypeError, NotImplementedError) as error:
                    raise DependencyError('extracted-function API unavailable: ' + str(error)) from error
                gc.collect()
                require(returned is None, 'extracted function returned error: ' + str(returned))
                require((store_dir / '.zarray').is_file(), 'persisted .zarray missing')
                current_inventory = inventory(inputs)
                require(input_inventory == current_inventory, 'TIFF inventory changed')
                after = [seal(Path(row['path'])) for row in fixtures]
                require(fixtures == after, 'TIFF pre/post seals')
                report['cases'].append({
                    'arm': arm, 'case': label, 'planes': planes, 'input_dtype': dtype,
                    'obytes': obytes, 'tiff_dir': str(inputs), 'store_dir': str(store_dir.resolve()),
                    'fixtures_before': fixtures, 'fixtures_after': after,
                    'store_files': inventory(store_dir), 'returned': returned})
            del function
            gc.collect()
        report['descriptors_after'] = {name: seal(path) for name, path in descriptors.items()}
        require(report['descriptors_before'] == report['descriptors_after'], 'descriptor pre/post seals')
        report['sources_before'] = before
        report['sources_after'] = {arm: seal(p) for arm, p in sources.items()}
        require(before == report['sources_after'], 'source pre/post seals')
        require(len(report['cases']) == 12, 'complete cohort')
        report['status'] = 'PRODUCER_COMPLETE_AWAITING_INDEPENDENT_CHECKER'
        code = 0
    except (DependencyError, ImportError, AttributeError) as error:
        report['status'] = 'UNKNOWN_STOP_DEPENDENCY_OR_API'
        report['failure'] = {'type': type(error).__name__, 'message': str(error),
                             'traceback': traceback.format_exc()}
    except Exception as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error),
                             'traceback': traceback.format_exc()}
    with (out / 'capture.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
