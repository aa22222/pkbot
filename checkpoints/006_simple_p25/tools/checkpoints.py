"""Create code-only checkpoints and stage them with the pinned shared asset."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOTS = ("baseline-bots", "docker", "tools")
ROOT_FILES = (".dockerignore", ".gitignore", "pytest.ini", "requirements-dev.txt", "requirements.txt")
CODE_SUFFIXES = {".py", ".sh", ".txt", ".ini", ".toml", ".yml", ".yaml", ".json", ".dockerignore"}
ASSET = Path("data/preflop.npy")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def files_to_copy():
    files = [p for p in ROOT.glob("*.py") if p.is_file()]
    files.extend(ROOT / name for name in ROOT_FILES if (ROOT / name).is_file())
    for dirname in CODE_ROOTS:
        files.extend(p for p in (ROOT / dirname).rglob("*")
                     if p.is_file() and (p.suffix.lower() in CODE_SUFFIXES or p.name.startswith("Dockerfile."))
                     and "__pycache__" not in p.parts)
    return sorted(set(files))


def safe_name(name):
    if not re.fullmatch(r"[0-9]{3}_[a-z0-9_]+", name):
        raise SystemExit("checkpoint name must look like 001_pair_tables")


def checkpoint(args):
    safe_name(args.name)
    destination = ROOT / "checkpoints" / args.name
    manifest_path = ROOT / "checkpoint-manifests" / f"{args.name}.json"
    if destination.exists() or manifest_path.exists():
        raise SystemExit(f"refusing to overwrite existing checkpoint: {args.name}")
    destination.mkdir(parents=True)
    entries = files_to_copy()
    for source in entries:
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    try:
        python_info = subprocess.check_output([sys.executable, "--version"], text=True).strip()
        import numpy
        numpy_version = numpy.__version__
    except Exception:
        python_info, numpy_version = sys.version, None
    manifest = {
        "checkpoint": args.name,
        "parent": args.parent,
        "status": args.status,
        "rationale": args.rationale,
        "source_hashes": {p.relative_to(ROOT).as_posix(): digest(p) for p in entries},
        "asset_hashes": {ASSET.as_posix(): digest(ROOT / ASSET)},
        "python": python_info,
        "numpy": numpy_version,
        "validation": args.validation,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created {destination} ({len(entries)} code/config files)")
    print(f"Manifest: {manifest_path}")


def stage(args):
    safe_name(args.name)
    source = ROOT / "checkpoints" / args.name
    destination = (ROOT / args.destination).resolve()
    if ROOT.resolve() not in destination.parents:
        raise SystemExit("staging destination must stay inside the workspace")
    if not source.is_dir():
        raise SystemExit(f"checkpoint does not exist: {source}")
    manifest_path = ROOT / "checkpoint-manifests" / f"{args.name}.json"
    if not manifest_path.is_file():
        raise SystemExit(f"checkpoint manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sources = manifest["source_hashes"]
    actual_sources = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()}
    if actual_sources != set(expected_sources):
        raise SystemExit("checkpoint file set does not match its manifest")
    for relative, expected in expected_sources.items():
        path = source / Path(relative)
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f"checkpoint source hash mismatch: {relative}")
    expected_asset = manifest["asset_hashes"][ASSET.as_posix()]
    if digest(ROOT / ASSET) != expected_asset:
        raise SystemExit(f"shared asset hash mismatch: {ASSET}")
    if destination.exists():
        raise SystemExit(f"refusing to overwrite staging directory: {destination}")
    destination.mkdir(parents=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    asset_target = destination / ASSET
    asset_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / ASSET, asset_target)
    print(f"Staged {args.name} with {ASSET} at {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    create = sub.add_parser("create", help="copy code/config into a new checkpoint")
    create.add_argument("name", help="monotonic name, e.g. 001_pair_tables")
    create.add_argument("--parent", required=True)
    create.add_argument("--status", default="experimental")
    create.add_argument("--rationale", default="")
    create.add_argument("--validation", default="pending primary review")
    create.set_defaults(run=checkpoint)
    stage_parser = sub.add_parser("stage", help="stage checkpoint plus pinned asset")
    stage_parser.add_argument("name")
    stage_parser.add_argument("--destination", default="external-bots/checkpoint_000")
    stage_parser.set_defaults(run=stage)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
