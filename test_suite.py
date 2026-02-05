"""CareFlow Test Suite Runner.

Runs all tests and outputs results to test_results.txt.
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.test_extraction import TestExtraction
from tests.test_reasoning import TestReasoning
from tests.test_booking import TestBooking
from tests.test_concept_query import TestConceptQuery
from tests.test_retrieval import TestRetrieval


def run_all_tests(output_file: str = "test_results.txt"):
    """Run all test suites and output results.

    Args:
        output_file: Path to output file for results
    """
    print("=" * 70)
    print("CAREFLOW TEST SUITE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    all_results = []
    total_passed = 0
    total_tests = 0

    # Run Extraction Tests
    print("Running Extraction Tests...")
    extraction_suite = TestExtraction()
    extraction_results = extraction_suite.run_all()
    all_results.append(extraction_results)
    total_passed += extraction_results["passed"]
    total_tests += extraction_results["total"]
    print(f"  -> {extraction_results['passed']}/{extraction_results['total']} passed")
    print()

    # Run Reasoning Tests
    print("Running Reasoning Tests...")
    reasoning_suite = TestReasoning()
    reasoning_results = reasoning_suite.run_all()
    all_results.append(reasoning_results)
    total_passed += reasoning_results["passed"]
    total_tests += reasoning_results["total"]
    print(f"  -> {reasoning_results['passed']}/{reasoning_results['total']} passed")
    print()

    # Run Booking Tests
    print("Running Booking Tests...")
    booking_suite = TestBooking()
    booking_results = booking_suite.run_all()
    all_results.append(booking_results)
    total_passed += booking_results["passed"]
    total_tests += booking_results["total"]
    print(f"  -> {booking_results['passed']}/{booking_results['total']} passed")
    print()

    # Run Concept Query Tests (PHI-Aware De-identification)
    print("Running Concept Query Tests...")
    concept_suite = TestConceptQuery()
    concept_results = concept_suite.run_all()
    all_results.append(concept_results)
    total_passed += concept_results["passed"]
    total_tests += concept_results["total"]
    print(f"  -> {concept_results['passed']}/{concept_results['total']} passed")
    print()

    # Run Retrieval Tests (Hybrid Vector Strategy)
    print("Running Retrieval Tests...")
    retrieval_suite = TestRetrieval()
    retrieval_results = retrieval_suite.run_all()
    all_results.append(retrieval_results)
    total_passed += retrieval_results["passed"]
    total_tests += retrieval_results["total"]
    print(f"  -> {retrieval_results['passed']}/{retrieval_results['total']} passed")
    print()

    # Print Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for suite_result in all_results:
        suite_name = suite_result["suite"]
        passed = suite_result["passed"]
        total = suite_result["total"]
        status = "PASS" if passed == total else "FAIL"
        print(f"  [{status}] {suite_name}: {passed}/{total}")

    print("-" * 70)
    overall_status = "ALL TESTS PASSED" if total_passed == total_tests else "SOME TESTS FAILED"
    print(f"  TOTAL: {total_passed}/{total_tests} ({overall_status})")
    print("=" * 70)

    # Write detailed results to file
    with open(output_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("CAREFLOW TEST RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total: {total_passed}/{total_tests} tests passed\n")
        f.write("\n")

        for suite_result in all_results:
            suite_name = suite_result["suite"]
            passed = suite_result["passed"]
            total = suite_result["total"]

            f.write("-" * 70 + "\n")
            f.write(f"{suite_name.upper()} TESTS ({passed}/{total})\n")
            f.write("-" * 70 + "\n")

            for test in suite_result["results"]:
                status = "PASS" if test["passed"] else "FAIL"
                f.write(f"  [{status}] {test['test']}\n")
                if test["details"]:
                    f.write(f"           {test['details']}\n")

            f.write("\n")

        # Summary
        f.write("=" * 70 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 70 + "\n")

        for suite_result in all_results:
            suite_name = suite_result["suite"]
            passed = suite_result["passed"]
            total = suite_result["total"]
            f.write(f"  {suite_name}: {passed}/{total}\n")

        f.write("-" * 70 + "\n")
        f.write(f"  TOTAL: {total_passed}/{total_tests}\n")

        if total_passed == total_tests:
            f.write("\n  STATUS: ALL TESTS PASSED\n")
        else:
            failed = total_tests - total_passed
            f.write(f"\n  STATUS: {failed} TESTS FAILED\n")

    print(f"\nDetailed results written to: {output_file}")

    return {
        "total_passed": total_passed,
        "total_tests": total_tests,
        "all_passed": total_passed == total_tests,
        "suites": all_results
    }


def print_detailed_results(all_results: list):
    """Print detailed test results to console."""
    print()
    print("=" * 70)
    print("DETAILED TEST RESULTS")
    print("=" * 70)

    for suite_result in all_results:
        suite_name = suite_result["suite"]
        print()
        print(f"--- {suite_name.upper()} TESTS ---")

        for test in suite_result["results"]:
            status = "PASS" if test["passed"] else "FAIL"
            print(f"  [{status}] {test['test']}")
            if test["details"]:
                print(f"         {test['details']}")


if __name__ == "__main__":
    results = run_all_tests()

    # Print detailed results
    print_detailed_results(results["suites"])

    # Exit with appropriate code
    sys.exit(0 if results["all_passed"] else 1)
