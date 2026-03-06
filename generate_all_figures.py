#!/usr/bin/env python3
"""
generate_all_figures.py
========================
Generate **all** publication-quality figures for the LSMCMC paper in one run.

This script sequentially executes:
  1. generate_paper_figures.py   – LG, MLSWE linear, NL twin, NL real figures
  2. generate_nlgamma_figures.py – arctan + Cauchy (non-Gaussian) figures
  3. regen_timeseries_1x2.py    – V2 velocity / SST / SSH timeseries (1×2 panels)

All figures are saved to  paper_figures/  (PDF + PNG).

Usage
-----
    python3 generate_all_figures.py
"""
import os
import sys
import time
import runpy

BASEDIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASEDIR)

SCRIPTS = [
    'generate_paper_figures.py',
    'generate_nlgamma_figures.py',
    'regen_timeseries_1x2.py',
]


def main():
    t0 = time.time()
    print("=" * 64)
    print("  generate_all_figures.py — LSMCMC paper figure generation")
    print("=" * 64)

    for script in SCRIPTS:
        path = os.path.join(BASEDIR, script)
        if not os.path.isfile(path):
            print(f"\n  [SKIP] {script} not found")
            continue

        print(f"\n{'─' * 64}")
        print(f"  Running {script} ...")
        print(f"{'─' * 64}")
        ts = time.time()
        try:
            runpy.run_path(path, run_name='__main__')
        except SystemExit:
            pass  # some scripts call sys.exit(0)
        except Exception as exc:
            print(f"  [ERROR] {script}: {exc}")
        elapsed = time.time() - ts
        print(f"  {script} finished in {elapsed:.1f}s")

    total = time.time() - t0
    print(f"\n{'=' * 64}")
    print(f"  All figures generated in {total:.1f}s")
    print(f"  Output directory: {os.path.join(BASEDIR, 'paper_figures')}")
    print(f"{'=' * 64}")


if __name__ == '__main__':
    main()
