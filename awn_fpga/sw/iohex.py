"""Shared helpers: int8/int32 numpy <-> hex text files for iverilog $readmemh."""
import os
import numpy as np


def write_int8_hex(path, arr):
    """Write a numpy int8 array as one hex byte per line (lower-cased, 2 chars).
    Reshapes to 1D, preserving row-major order."""
    a = np.asarray(arr, dtype=np.int8).reshape(-1)
    u = a.view(np.uint8)
    with open(path, "w") as f:
        for v in u:
            f.write(f"{int(v):02x}\n")


def read_int8_hex(path, count=None):
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(int(s, 16))
    a = np.array(out, dtype=np.uint8).view(np.int8)
    if count is not None:
        assert a.size == count, f"{path}: expected {count} elems, got {a.size}"
    return a


def write_int32_hex(path, arr):
    a = np.asarray(arr, dtype=np.int32).reshape(-1)
    u = a.view(np.uint32)
    with open(path, "w") as f:
        for v in u:
            f.write(f"{int(v):08x}\n")


def read_int32_hex(path, count=None):
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(int(s, 16))
    a = np.array(out, dtype=np.uint32).view(np.int32)
    if count is not None:
        assert a.size == count
    return a


def assert_equal_int8(actual, golden, name=""):
    actual = np.asarray(actual, dtype=np.int8).reshape(-1)
    golden = np.asarray(golden, dtype=np.int8).reshape(-1)
    if actual.size != golden.size:
        raise AssertionError(f"{name}: size {actual.size} vs {golden.size}")
    diff = np.where(actual != golden)[0]
    if diff.size:
        i = diff[0]
        raise AssertionError(
            f"{name}: first mismatch at idx {i}: hw={int(actual[i])} ref={int(golden[i])}, "
            f"{diff.size}/{actual.size} differ")


def assert_equal_int32(actual, golden, name=""):
    actual = np.asarray(actual, dtype=np.int32).reshape(-1)
    golden = np.asarray(golden, dtype=np.int32).reshape(-1)
    if actual.size != golden.size:
        raise AssertionError(f"{name}: size {actual.size} vs {golden.size}")
    diff = np.where(actual != golden)[0]
    if diff.size:
        i = diff[0]
        raise AssertionError(
            f"{name}: first mismatch at idx {i}: hw={int(actual[i])} ref={int(golden[i])}, "
            f"{diff.size}/{actual.size} differ")
