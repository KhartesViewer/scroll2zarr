import os
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, check_deadline, parser, read_json, require, source_seals, vmhwm, write_json
from runtime import environment


def main():
    cli = parser()
    cli.add_argument('--id', required=True)
    cli.add_argument('--role', required=True)
    cli.add_argument('--deadline', required=True, type=float)
    args = cli.parse_args()
    protocol = read_json(HERE / 'protocol.json')
    require({'id': args.id, 'role': args.role} in protocol['arms'], 'declared arm')
    producer, np, tifffile, zarr, identity, before = environment(args, protocol)
    binding = protocol['sources'][args.role]
    source = HERE / protocol['snapshots'][args.role]['relative']
    function = producer.extract(source, binding, np, tifffile, zarr)
    baseline = vmhwm()
    check_deadline(args.deadline)
    result = function(args.out / 'fixture', args.out / 'outputs' / args.id,
                      32, obytes=1, slices=None, maxgb=None)
    post = vmhwm()
    require(result is None, 'terminal non-None converter return ' + repr(result))
    check_deadline(args.deadline)
    require(source_seals() == before, 'arm source pre/post')
    write_json(args.out / (args.id + '.json'), {'status': 'CONVERTED', 'id': args.id,
               'role': args.role, 'pid': os.getpid(), 'returned_is_none': True,
               'post_import_vmhwm': baseline, 'post_conversion_vmhwm': post,
               'environment': identity, 'sources': before,
               'vmhwm_endpoint': 'immediately after conversion, before receipt and final source seals'})


if __name__ == '__main__':
    main()
