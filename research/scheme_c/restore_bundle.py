"""Verify and restore the release archive without replacing existing files.

python3 -B -m research.scheme_c.restore_bundle --archive /path/to/archive.tar.gz
Uses only the Python standard library; install research dependencies afterward.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

HERE = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def restore(archive, manifest, repo):
    archive, repo = Path(archive), Path(repo).resolve()
    if digest(archive) != manifest['sha256']:
        raise ValueError('Archive SHA-256 does not match bundle.json')
    expected = manifest['files']
    missing = []
    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError('Archive members differ from the manifest')
        # Validate all members and existing destinations before writing anything.
        for member in members:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts
                    or tuple(path.parts[:3]) not in {
                        ('research', 'scheme_c', 'inputs'), ('research', 'scheme_c', 'results')}):
                raise ValueError('Unsafe archive member: ' + member.name)
            dest = repo.joinpath(*path.parts)
            for parent in [dest] + list(dest.parents):
                if parent == repo:
                    break
                if parent.is_symlink():
                    raise ValueError('Refusing symlink destination: ' + member.name)
            record = expected[member.name]
            stream = bundle.extractfile(member)
            value = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                value.update(block)
            if member.size != record['size'] or value.hexdigest() != record['sha256']:
                raise ValueError('Member checksum mismatch: ' + member.name)
            if dest.exists():
                if not dest.is_file() or digest(dest) != record['sha256']:
                    raise ValueError('Existing file differs; nothing overwritten: ' + member.name)
            else:
                missing.append((member, dest))
        for member, dest in missing:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as output, bundle.extractfile(member) as source:
                shutil.copyfileobj(source, output)
    return {'files_verified': len(members), 'files_restored': len(missing),
            'existing_identical_files_preserved': len(members) - len(missing)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    args = parser.parse_args()
    manifest = json.loads((HERE / 'bundle.json').read_text())
    print(json.dumps(restore(args.archive, manifest, HERE.parents[1]), indent=2))
