"""Download LIBERO HDF5 demos from HuggingFace.

Default suite: libero_spatial (10 tasks). The HDF5 files live under
data/libero/<suite>/*.hdf5 after download.

Source: https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
"""
from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download

REPO_ID = "yifengzhu-hf/LIBERO-datasets"
DEFAULT_SUITES = ["libero_spatial"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    p.add_argument("--dest", default="data/libero")
    args = p.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    allow = [f"{s}/*" for s in args.suites]
    print(f"[download] repo={REPO_ID} suites={args.suites} -> {args.dest}")
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=allow,
        local_dir=args.dest,
    )
    print(f"[download] done -> {path}")
    for s in args.suites:
        sd = os.path.join(args.dest, s)
        if os.path.isdir(sd):
            n = len([f for f in os.listdir(sd) if f.endswith(".hdf5")])
            print(f"[download] {s}: {n} hdf5 files")


if __name__ == "__main__":
    main()
