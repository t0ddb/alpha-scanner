import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from research.scheme_c.protection import snapshot
from research.scheme_c.restore_bundle import digest, restore


class PackagingTests(unittest.TestCase):
    def archive(self, root, entries):
        archive = root / 'bundle.tar.gz'
        manifest = {'files': {}}
        with tarfile.open(archive, 'w:gz') as output:
            for name, data in entries:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                output.addfile(info, io.BytesIO(data))
                manifest['files'][name] = {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        manifest['sha256'] = digest(archive)
        return archive, manifest

    def test_restore_and_repeat_preserve_existing_identical_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, manifest = self.archive(root, [('research/scheme_c/inputs/a', b'input')])
            self.assertEqual(restore(archive, manifest, root)['files_restored'], 1)
            self.assertEqual(restore(archive, manifest, root)['files_restored'], 0)

    def test_conflicting_file_prevents_all_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, manifest = self.archive(root, [('research/scheme_c/inputs/a', b'new'),
                                                    ('research/scheme_c/inputs/b', b'old')])
            conflict = root / 'research/scheme_c/inputs/b'
            conflict.parent.mkdir(parents=True)
            conflict.write_bytes(b'user data')
            with self.assertRaisesRegex(ValueError, 'Existing file differs'):
                restore(archive, manifest, root)
            self.assertFalse((conflict.parent / 'a').exists())
            self.assertEqual(conflict.read_bytes(), b'user data')

    def test_bad_archive_hash_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, manifest = self.archive(root, [('research/scheme_c/inputs/a', b'input')])
            manifest['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'Archive SHA-256'):
                restore(archive, manifest, root)

    def test_traversal_rejected_even_if_listed_in_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, manifest = self.archive(root, [('research/scheme_c/inputs/../../../escape', b'input')])
            with self.assertRaisesRegex(ValueError, 'Unsafe archive member'):
                restore(archive, manifest, root)

    def test_symlink_destination_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, manifest = self.archive(root, [('research/scheme_c/inputs/a', b'input')])
            (root / 'research').symlink_to(root / 'elsewhere')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                restore(archive, manifest, root)

    def test_protection_uses_current_files_not_historical_backups(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'state.json').write_text('current state')
            (root / '.env').write_text('not inspected')
            before = snapshot(root)
            self.assertEqual(set(before), {'state.json'})
            (root / 'state.json').write_text('changed')
            self.assertNotEqual(before, snapshot(root))
            (root / 'state.json').unlink()
            self.assertNotEqual(before, snapshot(root))


if __name__ == '__main__':
    unittest.main()
