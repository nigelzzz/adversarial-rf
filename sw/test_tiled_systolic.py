#!/usr/bin/env python3
"""Wrapper that forwards to awn_fpga/sw/test_tiled_systolic.py from any CWD."""
import subprocess, sys, os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_SCRIPT = os.path.join(REPO_ROOT, 'awn_fpga', 'sw', 'test_tiled_systolic.py')
sys.exit(subprocess.call([sys.executable, REAL_SCRIPT] + sys.argv[1:]))
