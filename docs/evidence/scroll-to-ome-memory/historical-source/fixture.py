import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, check_deadline, fixture_seals, parser, read_json, require, source_seals, write_json
from runtime import environment


def main():
    cli = parser()
    cli.add_argument('--deadline', required=True, type=float)
    args = cli.parse_args()
    protocol = read_json(HERE / 'protocol.json')
    _, np, tifffile, _, identity, before = environment(args, protocol)
    directory = args.out / 'fixture'
    directory.mkdir()
    width = np.arange(2048, dtype=np.uint32)[None, :]
    for plane in range(32):
        check_deadline(args.deadline)
        def stripes():
            for start in range(0, 2048, 64):
                check_deadline(args.deadline)
                height = np.arange(start, start + 64, dtype=np.uint32)[:, None]
                yield ((257 * plane + 17 * height + 31 * width) % 65536).astype('<u2')
        with tifffile.TiffWriter(directory / ('%02d.tif' % plane), byteorder='<') as writer:
            writer.write(data=stripes(), shape=(2048, 2048), dtype='<u2', rowsperstrip=64,
                         compression=None, photometric='minisblack', metadata=None)
    seals = fixture_seals(args.out, protocol, args.deadline)
    require(source_seals() == before, 'fixture source pre/post')
    write_json(args.out / 'fixture.json', {'status': 'FIXTURE_SEALED', 'seals': seals,
               'environment': identity, 'sources': before})


if __name__ == '__main__':
    main()
