#!/usr/bin/env python3
"""Verify a local Hugging Face checkpoint directory against the Hub tree (stdlib only).

Compares file count, per-file size, and (with --sha256) the sha256 of every LFS file
against ``lfs.oid`` from the Hub tree API for a pinned revision. Writes SHA256SUMS
(usable with ``sha256sum -c`` on the other node) and verify.json into --out.

Usage:
  verify-checkpoint.py ~/models/GLM-5.3-Flash-NVFP4 --repo RedHatAI/GLM-5.3-Flash-NVFP4 \
      --revision <sha> [--sha256] [--out ~/glm53-cluster/results]
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request


def hub_tree(repo: str, revision: str) -> list[dict]:
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true"
    req = urllib.request.Request(url, headers={"User-Agent": "verify-checkpoint/1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("local_dir")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument(
        "--sha256",
        action="store_true",
        help="hash every LFS file (slow, ~15-25 min for 181 GiB)",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    entries = [e for e in hub_tree(args.repo, args.revision) if e.get("type") == "file"]
    expected_total = sum(e.get("size", 0) for e in entries)
    problems = []
    sums = []
    local_total = 0
    for e in entries:
        rel = e["path"]
        p = os.path.join(args.local_dir, rel)
        if not os.path.isfile(p):
            problems.append(f"missing: {rel}")
            continue
        size = os.path.getsize(p)
        local_total += size
        if size != e.get("size", size):
            problems.append(f"size mismatch: {rel} local={size} hub={e.get('size')}")
        lfs = e.get("lfs") or {}
        if args.sha256 and lfs.get("oid"):
            digest = sha256_of(p)
            sums.append(f"{digest}  {rel}")
            if digest != lfs["oid"]:
                problems.append(f"sha256 mismatch: {rel}")
            print(
                f"  ok {rel}" if digest == lfs["oid"] else f"  BAD {rel}",
                file=sys.stderr,
            )

    report = {
        "repo": args.repo,
        "revision": args.revision,
        "files_expected": len(entries),
        "bytes_expected": expected_total,
        "bytes_local": local_total,
        "lfs_files": sum(1 for e in entries if (e.get("lfs") or {}).get("oid")),
        "sha256_checked": bool(args.sha256),
        "problems": problems,
    }
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "verify.json"), "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        if sums:
            with open(os.path.join(args.out, "SHA256SUMS"), "w") as f:
                f.write("\n".join(sums) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
