#!/usr/bin/env python3
"""Flatten a YAML config (with `include:` chain) into argparse-style CLI flags
suitable for `python -m main` / `torchrun ... main.py`.

Output format: one `--key value` per token, all on one line, ready for:

    args=$(python scripts/infra/yaml_to_args.py configs/imagenet/convnext_tiny.yaml)
    torchrun ... main.py $args --output_dir ...

Boolean values are passed verbatim ("true"/"false") because main.py uses
str2bool. None / null become the literal string "None" only when the
upstream argparse default is None — otherwise the key is omitted.

Designed to be a one-screen utility, not a YAML schema validator: an
unknown key just becomes `--unknown_key value` and main.py will reject
it loudly.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def _load_with_includes(path: Path, _seen: set[Path] | None = None) -> dict:
    """Load YAML, then recursively merge any `include:` parent.

    Child wins on key collision (override semantics). `include:` is a
    path relative to the file declaring it. Cycle-safe.
    """
    path = path.resolve()
    if _seen is None:
        _seen = set()
    if path in _seen:
        raise ValueError(f"include cycle through {path}")
    _seen.add(path)

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    parent = cfg.pop("include", None)
    if parent is not None:
        parent_path = (path.parent / parent).resolve()
        merged = _load_with_includes(parent_path, _seen)
        merged.update(cfg)
        cfg = merged
    return cfg


def _emit(key: str, value) -> list[str]:
    """Render one (key, value) into argparse-style tokens."""
    flag = f"--{key}"
    if value is None:
        # main.py treats absence as default; explicit None is rarely useful.
        return []
    if isinstance(value, bool):
        # main.py uses str2bool; pass true/false literally.
        return [flag, "true" if value else "false"]
    if isinstance(value, (list, tuple)):
        # `--opt_betas 0.9 0.999` style (nargs='+').
        return [flag, *(str(v) for v in value)]
    return [flag, str(value)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path, help="path to YAML config")
    p.add_argument(
        "--shell-quote",
        action="store_true",
        help="quote each token for safe inclusion in a shell command line",
    )
    args = p.parse_args()

    cfg = _load_with_includes(args.config)

    tokens: list[str] = []
    for k, v in cfg.items():
        tokens.extend(_emit(k, v))

    if args.shell_quote:
        import shlex
        print(" ".join(shlex.quote(t) for t in tokens))
    else:
        print(" ".join(tokens))
    return 0


if __name__ == "__main__":
    sys.exit(main())
