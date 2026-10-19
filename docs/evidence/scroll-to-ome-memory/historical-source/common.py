import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

HERE = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n')


def seal(path):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink forbidden')
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'regular unshared file')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            digest.update(block)
    return {'bytes': info.st_size, 'sha256': digest.hexdigest()}


def source_seals():
    frozen = read_json(HERE / 'source-freeze.json')
    observed = {}
    actual = {p.relative_to(HERE).as_posix() for p in HERE.rglob('*') if p.is_file()}
    require(actual == set(frozen['files']) | {'source-freeze.json'}, 'exact capsule cohort')
    for relative, expected in frozen['files'].items():
        path = HERE / relative
        require(path.resolve().is_relative_to(HERE), 'relative source escape')
        observed[relative] = seal(path)
        require(observed[relative] == expected, 'source identity ' + relative)
    observed['source-freeze.json'] = seal(HERE / 'source-freeze.json')
    return observed


def check_deadline(deadline):
    require(time.monotonic() < deadline, 'absolute work deadline')


def vmhwm():
    for line in Path('/proc/self/status').read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == 'VmHWM:':
            require(len(fields) == 3 and fields[2] == 'kB', 'Linux VmHWM units')
            raw = int(fields[1])
            require(raw > 0, 'positive VmHWM')
            return {'raw': raw, 'raw_unit': 'kB', 'multiplier': 1024, 'bytes': raw * 1024,
                    'approximate': True, 'observed_monotonic': time.monotonic()}
    raise ValueError('VmHWM missing')


def parser():
    result = argparse.ArgumentParser()
    for option in ('out', 'interpreter', 'isolated-env', 'environment-lock'):
        result.add_argument('--' + option, required=True, type=Path)
    return result


def check_selection(args, protocol):
    selection = read_json(HERE / 'runtime-selection.json')
    require(selection['status'] == 'BOUND_FOR_REVIEW', 'NOT_READY_RUNTIME_SELECTION')
    for name in ('out', 'interpreter', 'isolated_env', 'environment_lock'):
        value = getattr(args, name)
        require(value.is_absolute(), 'absolute CLI path ' + name)
        require(not any(p.is_symlink() for p in value.parents), 'symlink parent ' + name)
    for name in ('interpreter', 'isolated_env', 'environment_lock'):
        require(str(getattr(args, name)) == selection[name], 'selected route ' + name)
    require(seal(args.environment_lock)['sha256'] == selection['environment_lock_sha256'],
            'selected lock identity')
    require(selection['python'][:2] == [3, 12], 'selected Python3.12 required')
    return selection


def fixture_seals(out, protocol, deadline):
    directory = out / 'fixture'
    require(sorted(p.name for p in directory.iterdir()) == protocol['fixture']['names'],
            'exact numeric TIFF cohort')
    result = {}
    for name in protocol['fixture']['names']:
        check_deadline(deadline)
        result[name] = seal(directory / name)
    return result


def tree_sizes(out, *, live=False, entry_limit=20000, file_limit=12582912):
    totals = {'bulk': 0, 'log': 0, 'metadata': 0}
    pending = [Path(out)]
    entries = 1
    while pending:
        directory = pending.pop()
        try:
            directory_info = directory.lstat()
            require(stat.S_ISDIR(directory_info.st_mode), 'output directory type')
            with os.scandir(directory) as scan:
                paths = [Path(entry.path) for entry in scan]
        except FileNotFoundError as error:
            if live and error.errno == errno.ENOENT and directory != Path(out):
                continue
            raise
        for path in paths:
            entries += 1
            require(entries <= entry_limit, 'output entry bound')
            try:
                info = path.lstat()
            except FileNotFoundError as error:
                if live and error.errno == errno.ENOENT:
                    continue
                raise
            require(not stat.S_ISLNK(info.st_mode), 'output symlink')
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
                continue
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'regular unshared output file')
            require(info.st_size <= file_limit, 'output per-file byte bound')
            relative = path.relative_to(out)
            if path.suffix in ('.stdout', '.stderr'):
                category = 'log'
            elif relative.parts[0] == 'fixture' or (relative.parts[0] == 'outputs' and not path.name.startswith('.')):
                category = 'bulk'
            else:
                category = 'metadata'
            totals[category] += info.st_size
    return totals


def bounded_sizes(out, protocol, *, live=False):
    limits = protocol['prospective_limits']
    totals = tree_sizes(out, live=live, entry_limit=limits['output_entries'], file_limit=limits['file_bytes'])
    for category, count in totals.items():
        require(count <= limits[category + '_bytes'], category + ' byte bound')
    require(sum(totals.values()) <= limits['bulk_bytes'], 'aggregate output byte bound')
    return totals


def outcome_exit(status):
    return {'ACCEPT_WITNESS': 0, 'REJECT_WITNESS': 1,
            'REJECT_WITNESS_CORRECTNESS': 1, 'INCONCLUSIVE': 2}.get(status, 2)


def memory_verdict(rows, protocol):
    expected = protocol['arms']
    try:
        require(len(rows) == 4, 'four arms')
        supervisor = []
        baselines = []
        metrics = {'wait4': {}, 'vmhwm': {}}
        for row, spec in zip(rows, expected):
            require(row['id'] == spec['id'] and row['role'] == spec['role'], 'ABBA cohort')
            require(row['exit_code'] == 0 and not row['timed_out'], 'complete arm')
            own = row['supervisor_vmhwm']
            require(own['raw_unit'] == 'kB' and own['bytes'] == own['raw'] * 1024, 'supervisor units')
            supervisor.append(own['bytes'])
            wait = row['usage']
            child = row['arm_receipt']
            require(child['pid'] == row['pid'] and child['id'] == spec['id'] and child['role'] == spec['role'], 'PID binding')
            require(child['returned_is_none'] is True and child['status'] == 'CONVERTED', 'converter completion')
            require(wait['ru_maxrss_unit'] == 'KiB' and wait['ru_maxrss_bytes'] == wait['raw']['ru_maxrss'] * 1024, 'wait4 units')
            hwm = child['post_conversion_vmhwm']
            require(hwm['raw_unit'] == 'kB' and hwm['bytes'] == hwm['raw'] * 1024, 'arm units')
            baseline = child['post_import_vmhwm']
            require(baseline['bytes'] > 0 and baseline['bytes'] <= hwm['bytes'], 'HWM sequence')
            baselines.append(baseline['bytes'])
            for name, value in (('wait4', wait['ru_maxrss_bytes']), ('vmhwm', hwm['bytes'])):
                require(type(value) is int and value > 0, 'positive metric')
                metrics[name].setdefault(spec['role'], []).append(value)
        require(max(supervisor) < min(baselines + [v for roles in metrics.values() for values in roles.values() for v in values]), 'supervisor below all child observations')
        decisions = {}
        threshold = protocol['thresholds']
        for name, roles in metrics.items():
            reference = min(roles['reference_fixed'])
            candidate = max(roles['candidate'])
            passed = (reference - candidate >= threshold['absolute_bytes'] and
                      threshold['ratio_denominator'] * candidate <= threshold['ratio_numerator'] * reference)
            decisions[name] = {'reference_min_bytes': reference, 'candidate_max_bytes': candidate,
                               'meets_frozen_threshold': passed}
        return {'status': 'ACCEPT_WITNESS' if all(d['meets_frozen_threshold'] for d in decisions.values()) else 'REJECT_WITNESS',
                'metrics': decisions, 'scope': 'one fixed synthetic workload; different measurement endpoints'}
    except (KeyError, TypeError, ValueError) as error:
        return {'status': 'INCONCLUSIVE', 'reason': str(error)}
