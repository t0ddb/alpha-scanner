"""Read-only fingerprints for files present in the checkout running a study."""
import hashlib
from pathlib import Path


def snapshot(repo):
    repo = Path(repo)
    # Include current root state/config/code and workflows. Do not read secrets
    # or require historical ignored database copies to exist on another machine.
    paths = [p for p in repo.iterdir() if p.is_file() and not p.name.startswith('.')]
    paths += [p for p in (repo / '.github/workflows').glob('*') if p.is_file()]
    return {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}
