#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run all model tests and report results"""
import subprocess
import sys
import os
import time

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TEST_DIR)

tests = [
    "test_fsmn_vad.py",
    "test_ct_transformer.py",
    "test_paraformer.py",
    "test_sensevoice.py",
    "test_campplus.py",
    "test_paraformer_streaming.py",
    "test_qwen3_asr.py",
    "test_glm_asr.py",
]

SEP = "-" * 60
DSEP = "=" * 60

def main():
    results = {}
    total_start = time.time()

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_DIR + ":" + env.get("PYTHONPATH", "")

    print(DSEP)
    print("FunASR Model Tests")
    print(DSEP)

    for test_file in tests:
        test_path = os.path.join(TEST_DIR, test_file)
        print("\n" + SEP)
        print("Running: " + test_file)
        print(SEP)

        t0 = time.time()
        try:
            result = subprocess.run(
                [sys.executable, test_path],
                cwd=PROJECT_DIR,
                env=env,
                timeout=300,
                capture_output=False,
            )
            elapsed = time.time() - t0
            results[test_file] = ("PASSED" if result.returncode == 0 else "FAILED", elapsed)
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            results[test_file] = ("TIMEOUT", elapsed)
            print("  TIMEOUT after %.1fs" % elapsed)
        except Exception as e:
            elapsed = time.time() - t0
            results[test_file] = ("ERROR", elapsed)
            print("  ERROR: %s" % e)

    total_elapsed = time.time() - total_start

    print("\n" + DSEP)
    print("SUMMARY")
    print(DSEP)
    passed = 0
    failed = 0
    for test_file, (status, elapsed) in results.items():
        icon = "+" if status == "PASSED" else "x"
        print("  %s %-35s %-8s (%.1fs)" % (icon, test_file, status, elapsed))
        if status == "PASSED":
            passed += 1
        else:
            failed += 1

    print("\n  Total: %d passed, %d failed, %.1fs elapsed" % (passed, failed, total_elapsed))
    print(DSEP)
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
