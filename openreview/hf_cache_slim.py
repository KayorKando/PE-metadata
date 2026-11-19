"""Free /work quota: delete weight formats our code never loads from the Hugging Face cache (ONNX, OpenVINO,
Flax, TensorFlow, Rust). Everything here loads PyTorch weights (safetensors / pytorch_model.bin) only.

Only model dirs that really live in this cache are touched; dirs that are symlinks into another cache (e.g.
gpt2-large linked from /work/$USER/hf, another project's) are skipped. Dry run by default.

    python openreview/hf_cache_slim.py                       # list what would go, and the total
    python openreview/hf_cache_slim.py --delete              # delete it
"""
import argparse
import os
from pathlib import Path

DROP = ("onnx/", "openvino/", "flax_model.msgpack", "tf_model.h5", "rust_model.ot", "model.ot")


def unused_files(hub):
    """-> [(repo, file in snapshot, blob path or None, size)] for the formats in DROP."""
    out = []
    for repo in sorted(Path(hub).glob("models--*")):
        if repo.is_symlink():
            continue
        for snap in sorted((repo / "snapshots").glob("*")):
            for root, _, files in os.walk(snap):
                for f in files:
                    p = Path(root) / f
                    rel = p.relative_to(snap).as_posix()
                    if any(rel.startswith(d) if d.endswith("/") else rel.endswith(d) for d in DROP):
                        blob = Path(os.path.realpath(p)) if p.is_symlink() else None
                        size = blob.stat().st_size if blob is not None and blob.exists() else (
                            p.stat().st_size if not p.is_symlink() else 0)
                        out.append((repo.name, p, blob, size))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hub", default=str(Path(os.environ.get("HF_HOME", f"/work/{os.environ.get('USER', '')}/hf_cache"))
                                        / "hub"))
    p.add_argument("--delete", action="store_true")
    args = p.parse_args()
    files = unused_files(args.hub)
    total = 0
    for repo, f, blob, size in files:
        total += size
        print(f"{size / 1e6:9.1f} MB  {repo}  {f.name if f.parent.name not in ('onnx', 'openvino') else f.parent.name + '/' + f.name}")
        if args.delete:
            if blob is not None and blob.exists():
                blob.unlink()
            f.unlink()
    print(f"{'deleted' if args.delete else 'would delete (rerun with --delete)'}: {total / 1e9:.2f} GB in {len(files)} files")


if __name__ == "__main__":
    main()
