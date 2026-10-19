import os
import json
from pathlib import Path
import signal
import subprocess
import sys
sys.dont_write_bytecode = True
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))


def launch(script, extra, args, protocol, work_deadline, receipts, arm=None):
    common.check_deadline(work_deadline)
    name = arm['id'] if arm else script.removesuffix('.py')
    env = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'PYTHONHASHSEED': '0',
           'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1', **protocol['thread_env']}
    argv = [str(args.interpreter), '-I', '-B', str(common.HERE / script), '--out', str(args.out),
            '--interpreter', str(args.interpreter), '--isolated-env', str(args.isolated_env),
            '--environment-lock', str(args.environment_lock)]
    supervisor = common.vmhwm() if arm else None
    started = time.monotonic()
    deadline = min(work_deadline, started + 120) if arm else work_deadline
    argv += ['--deadline', repr(deadline), *extra]
    failure = None
    timed_out = False
    stopped_reason = None
    with (args.out / (name + '.stdout')).open('xb') as stdout, (args.out / (name + '.stderr')).open('xb') as stderr:
        child = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env=env,
                                 stdin=subprocess.DEVNULL, start_new_session=True, cwd=args.out)
        last_scan = started
        while True:
            pid, status, usage = os.wait4(child.pid, os.WNOHANG)
            if pid:
                reaped = time.monotonic()
                child.returncode = os.waitstatus_to_exitcode(status)
                timed_out = timed_out or reaped >= deadline
                if timed_out and stopped_reason is None:
                    stopped_reason = 'deadline'
                break
            now = time.monotonic()
            if failure is None and now - last_scan >= 0.5:
                try:
                    common.bounded_sizes(args.out, protocol, live=True)
                except Exception as error:
                    failure = str(error)
                last_scan = now
            if (now >= deadline or failure) and stopped_reason is None:
                timed_out = now >= deadline
                stopped_reason = 'deadline' if timed_out else 'size_guard'
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(0.02)
    raw = {key: getattr(usage, key) for key in dir(usage) if key.startswith('ru_')}
    row = {'id': name, 'pid': child.pid, 'argv': argv, 'exit_code': child.returncode,
           'raw_wait_status': status, 'started_before_popen_monotonic': started,
           'reaped_monotonic': reaped, 'deadline_monotonic': deadline,
           'elapsed_before_popen_to_reap_seconds': reaped - started, 'timed_out': timed_out,
           'failure': failure, 'stopped_reason': stopped_reason, 'usage': {'raw': raw, 'ru_maxrss_unit': 'KiB',
           'ru_maxrss_bytes': usage.ru_maxrss * 1024},
           'stdout': name + '.stdout', 'stderr': name + '.stderr'}
    if arm:
        row.update(role=arm['role'], supervisor_vmhwm=supervisor)
    receipts.append(row)
    common.write_json(args.out / (name + '-wait4.json'), row)
    allowed = (0, 1) if script == 'checker.py' else (0,)
    common.require(child.returncode in allowed and not timed_out and failure is None, 'terminal process failure ' + name)
    common.check_deadline(work_deadline)
    common.bounded_sizes(args.out, protocol)
    return row


def assay(report, context):
    started = time.monotonic()
    args = common.parser().parse_args()
    protocol = common.read_json(common.HERE / 'protocol.json')
    before = common.source_seals()
    common.require(sys.platform == 'linux', 'Linux wait4/RSS semantics required')
    common.check_selection(args, protocol)
    common.require(args.out.is_absolute() and not args.out.exists(), 'absolute fresh bulk output')
    deadline = started + protocol['prospective_limits']['work_seconds']
    args.out.mkdir()
    context['out'] = args.out
    (args.out / 'outputs').mkdir()
    report.update(protocol_status='SOURCE_PROPOSAL_UNEXECUTED_AT_AUTHORING',
                  work_started_monotonic=started, work_deadline_monotonic=deadline,
                  sources_before=before, processes=[], arms=[])
    launch('fixture.py', [], args, protocol, deadline, report['processes'])
    fixture = common.read_json(args.out / 'fixture.json')
    common.require(fixture['status'] == 'FIXTURE_SEALED' and fixture['sources'] == before, 'fixture seals first')
    frozen_tiffs = fixture['seals']
    for arm in protocol['arms']:
        common.require(common.fixture_seals(args.out, protocol, deadline) == frozen_tiffs, 'TIFFs before arm/cache preparation')
        row = launch('arm.py', ['--id', arm['id'], '--role', arm['role']], args,
                     protocol, deadline, report['processes'], arm)
        row['arm_receipt'] = common.read_json(args.out / (arm['id'] + '.json'))
        common.require(row['arm_receipt']['environment'] == fixture['environment'], 'identical selected environment')
        report['arms'].append(row)
        common.require(common.fixture_seals(args.out, protocol, deadline) == frozen_tiffs, 'TIFFs after arm')
    checker_row = launch('checker.py', [], args, protocol, deadline, report['processes'])
    checked = common.read_json(args.out / 'checker.json')
    common.require(checked['sources'] == before and checked['environment'] == fixture['environment'] and
            checked['fixture_post_seals'] == frozen_tiffs and checked['fixture_valid'] is True,
            'checker source/environment/valid fixture binding')
    if checked['status'] == 'REJECT_WITNESS_CORRECTNESS':
        common.require(checker_row['exit_code'] == 1, 'observed checker rejection exit')
        report.update(status='REJECT_WITNESS_CORRECTNESS', checker_failure=checked)
    else:
        common.require(checker_row['exit_code'] == 0 and checked['status'] == 'CHECKED_ALL_COORDINATES' and
                checked['coordinates_per_arm'] == 134217728 and
                set(checked['output_seals']) == {arm['id'] for arm in protocol['arms']},
                'independent checker completion/coverage')
        report['memory'] = common.memory_verdict(report['arms'], protocol)
        report['status'] = report['memory']['status']
    after = common.source_seals()
    common.require(after == before, 'final source pre/post seals')
    report['sources_after'] = after
    common.require(common.fixture_seals(args.out, protocol, deadline) == frozen_tiffs, 'final TIFF seals')
    report['checker_receipt'] = common.seal(args.out / 'checker.json')
    common.check_deadline(deadline)
    report['sizes_before_final_receipt'] = common.bounded_sizes(args.out, protocol)
    report['finished_monotonic'] = time.monotonic()
    common.write_json(args.out / 'assay.json', report)
    code = common.outcome_exit(report['status'])
    final_sizes = common.bounded_sizes(args.out, protocol)
    assay_identity = common.seal(args.out / 'assay.json')
    common.check_deadline(deadline)
    common.write_json(args.out / 'completion.json', {'status': report['status'], 'sizes_before_completion': final_sizes,
               'assay_receipt': assay_identity, 'checker_receipt': report['checker_receipt'],
               'finished_monotonic': time.monotonic(), 'requires_driver_exit_code': code,
               'authority': 'VALID_ONLY_WITH_MATCHING_ACTUAL_OBSERVED_DRIVER_EXIT_AND_PARENT_ACCEPTANCE'})
    common.bounded_sizes(args.out, protocol)
    common.check_deadline(deadline)
    return code


def main():
    global common
    report = {'status': 'INCONCLUSIVE'}
    context = {}
    try:
        import common
        return assay(report, context)
    except BaseException as error:
        report.update(status='INCONCLUSIVE', failure={'type': type(error).__name__, 'message': str(error)})
        out = context.get('out')
        if out is not None:
            try:
                common.write_json(out / 'assay.json', report)
            except BaseException:
                pass
        try:
            sys.stderr.write(json.dumps(report['failure']) + '\n')
        except BaseException:
            pass
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
