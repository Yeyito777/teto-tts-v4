#!/usr/bin/env python3
"""Backward-compatible wrapper; use fish_s2_infer.py instead."""
from fish_s2_infer import main

if __name__ == "__main__":
    raise SystemExit(main())
