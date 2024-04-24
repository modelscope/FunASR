#!/usr/bin/env python

import argparse
import os
import sys
import unittest
from fnmatch import fnmatch


def gather_test_cases(test_dir, pattern, list_tests):
    case_list = []
    for dirpath, dirnames, filenames in os.walk(test_dir):
        for file in filenames:
            if fnmatch(file, pattern):
                case_list.append(file)

    test_suite = unittest.TestSuite()

    for case in case_list:
        test_case = unittest.defaultTestLoader.discover(start_dir=test_dir, pattern=case)
        test_suite.addTest(test_case)
        if hasattr(test_case, "__iter__"):
            for subcase in test_case:
                if list_tests:
                    print(subcase)
        else:
            if list_tests:
                print(test_case)
    return test_suite


def main(args):
    runner = unittest.TextTestRunner()
    test_suite = gather_test_cases(os.path.abspath(args.test_dir), args.pattern, args.list_tests)
    if not args.list_tests:
        result = runner.run(test_suite)
        if len(result.failures) > 0:
            sys.exit(len(result.failures))
        if len(result.errors) > 0:
            sys.exit(len(result.errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("test runner")
    parser.add_argument("--list_tests", action="store_true", help="list all tests")
    parser.add_argument("--pattern", default="test_*.py", help="test file pattern")
    parser.add_argument("--test_dir", default="tests", help="directory to be tested")
    parser.add_argument("--disable_profile", action="store_true", help="disable profiling")
    args = parser.parse_args()
    print(f"working dir: {os.getcwd()}")
    main(args)
