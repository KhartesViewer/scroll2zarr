import importlib.util
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from common import HERE, check_selection, require, seal, source_seals


def environment(args, protocol):
    before = source_seals()
    selection = check_selection(args, protocol)
    require(list(sys.version_info[:3]) == selection['python'] and sys.version_info[:2] == (3, 12), 'selected Python3.12')
    require(sys.byteorder == 'little', 'native little-endian host required')
    require({k: os.environ.get(k) for k in protocol['thread_env']} == protocol['thread_env'], 'single-thread environment')
    helper = HERE / protocol['snapshots']['producer']['relative']
    spec = importlib.util.spec_from_file_location('frozen_producer', helper)
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    np, tifffile, zarr, deps = producer.check_environment(args, protocol)
    import numcodecs
    numcodecs.blosc.set_nthreads(1)
    identity = {'interpreter': str(Path(sys.executable).resolve()), 'version': sys.version,
                'prefix': str(Path(sys.prefix).resolve()), 'lock': seal(args.environment_lock),
                'dependencies': deps, 'thread_env': dict(protocol['thread_env']),
                'numcodecs_blosc_threads': numcodecs.blosc.get_nthreads()}
    require(identity['numcodecs_blosc_threads'] == 1, 'Blosc thread setting')
    return producer, np, tifffile, zarr, identity, before
