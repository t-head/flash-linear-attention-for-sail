# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Run FLA PR tests from a caselist, generating a JUnit XML report.

Adapted from FlashMLA-for-sail's pr-flashmla-ut.py.  Each non-empty,
non-comment line in the caselist is executed as a shell command in a fresh
subprocess. The runner records pytest's exit status and output, then writes
test-results.xml. If any case fails or times out, the process exits with code 1.
"""

import argparse
import contextlib
import os
import re
import signal
import subprocess
import sys
import time

from junit_xml import TestCase, TestSuite

OUTPUT_XML = "test-results.xml"

FAILURES_HEADER = re.compile(r"^=+\s*FAILURES\s*=+$")
SECTION_HEADER = re.compile(r"^=+ .+ =+$")
VERDICT_LINE = re.compile(r"^(FAILED|ERROR)\s+\S+")
DETAIL_LIMIT = 200


def derive_test_name(case_command: str) -> str:
    return re.sub(r"\s+", "_", case_command.strip())


def extract_failure_detail(output: str, limit: int = DETAIL_LIMIT) -> list[str]:
    """Return pytest's FAILURES section plus short-summary verdict lines.

    pytest prints tracebacks mid-output and ends with duration tables, so a
    plain tail window misses the actual failure.  Returns an empty list when
    the output carries no recognizable failure section (crash/timeout paths).
    """
    picked: list[str] = []
    in_section = False
    for line in output.splitlines():
        stripped = line.strip()
        if FAILURES_HEADER.match(stripped):
            in_section = True
        elif in_section and SECTION_HEADER.match(stripped):
            in_section = False
            continue
        if in_section or VERDICT_LINE.match(stripped):
            picked.append(line)
    if len(picked) > limit:
        omitted = len(picked) - limit
        picked = picked[:limit]
        picked.append(f"... ({omitted} more failure lines omitted)")
    return picked


def run_one_case(i: int, case_command: str, total_count: int, timeout: float = 1200, budget_note: str = ""):
    """Run one caselist command; return (junit_case, status, reason, output).

    status is one of PASS/FAIL/TIMEOUT/ERROR and is always printed as
    '>>> Result: [<status>]' so log search can find every verdict.
    """
    test_name = derive_test_name(case_command)
    junit_case = TestCase(name=test_name, classname="flaTest")

    print(f"\n[{i}/{total_count}] Running: {test_name}{budget_note}")
    print(f"Command: {case_command}")
    print("-" * 50)

    start_time = time.time()
    stdout_output: list[str] = []

    def emit(output: str) -> str:
        if output:
            print(output, end="")
            stdout_output.append(output)
        return "".join(stdout_output)

    try:
        process = subprocess.Popen(
            case_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )

        output, _ = process.communicate(timeout=timeout)
        collected = emit(output)
        duration = time.time() - start_time
        junit_case.elapsed_sec = duration

        junit_case.stdout = output
        # Pytest uses nonzero statuses for failures, collection errors and no tests.
        if process.returncode == 0:
            print(">>> Result: [PASS]")
            return junit_case, "PASS", "", collected

        reason = f"Exit code: {process.returncode}"
        print(f">>> Result: [FAIL] ({reason})")
        junit_case.add_failure_info(message=reason, output=collected)
        return junit_case, "FAIL", reason, collected

    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        collected = emit(output)
        junit_case.elapsed_sec = time.time() - start_time
        reason = f"Timeout after {timeout:g}s"
        print(f">>> Result: [TIMEOUT] ({reason})")
        junit_case.add_failure_info(message=reason, output=collected or "Test timed out")
        return junit_case, "TIMEOUT", reason, collected

    except Exception as e:
        junit_case.elapsed_sec = time.time() - start_time
        reason = f"Exception: {e}"
        print(f">>> Result: [ERROR] ({reason})")
        junit_case.add_error_info(message=reason, output=str(e))
        return junit_case, "ERROR", reason, "".join(stdout_output)


def parse_commands(filepath: str) -> list[str]:
    commands: list[str] = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            commands.append(line)
    return commands


def write_xml(xml_path: str, cases: list[TestCase]) -> None:
    """Persist the JUnit report so partial results survive external kills."""
    os.makedirs(os.path.dirname(xml_path) or ".", exist_ok=True)
    with open(xml_path, "w", encoding="utf-8") as f:
        TestSuite.to_file(f, [TestSuite("FLA PR Tests", cases)], prettyprint=True)


def run_cases(caselist_path: str, build_dir: str, output_xml: str = OUTPUT_XML, timeout: float = 1200, deadline: float = 0):
    if not os.path.exists(caselist_path):
        print(f"Error: caselist not found: {caselist_path}")
        sys.exit(1)

    test_commands = parse_commands(caselist_path)
    if not test_commands:
        print("No valid test commands found")
        sys.exit(1)

    os.chdir(build_dir)
    print(f"Working directory: {os.getcwd()}")
    budget = f"{deadline:g}s" if deadline else "disabled"
    print(f"Budget: per-case timeout {timeout:g}s; overall deadline {budget}")

    xml_path = os.path.abspath(output_xml)
    all_cases: list[TestCase] = []
    results: list[tuple[str, str, str, str, float]] = []
    failures: list[tuple[str, str, str, str, float]] = []
    passed = 0
    start_time = time.time()

    for i, cmd in enumerate(test_commands, 1):
        name = derive_test_name(cmd)
        elapsed = time.time() - start_time
        if deadline and elapsed > deadline:
            junit_case = TestCase(name=name, classname="flaTest")
            reason = f"Not run: overall deadline {deadline:g}s exceeded after {elapsed:.0f}s"
            print(f"\n[{i}/{len(test_commands)}] Skipped: {name}")
            print(f"    {reason}")
            junit_case.add_skipped_info(reason)
            all_cases.append(junit_case)
            results.append(("SKIPPED", name, reason, "", 0.0))
            failures.append(("SKIPPED", name, reason, "", 0.0))
            write_xml(xml_path, all_cases)
            continue
        junit_case, status, reason, output = run_one_case(
            i, cmd, len(test_commands), timeout, f" (elapsed {elapsed:.0f}s / {budget})"
        )
        duration = junit_case.elapsed_sec or 0.0
        all_cases.append(junit_case)
        results.append((status, name, reason, output, duration))
        if status == "PASS":
            passed += 1
        else:
            failures.append((status, name, reason, output, duration))
        write_xml(xml_path, all_cases)

    total = len(test_commands)
    elapsed_total = time.time() - start_time
    counts = {s: sum(1 for r in results if r[0] == s) for s in ("FAIL", "TIMEOUT", "ERROR", "SKIPPED")}
    print(
        f"\nTests completed. Total: {total}, Passed: {passed}, Failed: {counts['FAIL']}, "
        f"Timeout: {counts['TIMEOUT']}, Error: {counts['ERROR']}, Skipped: {counts['SKIPPED']}"
    )
    print(f"Elapsed: {elapsed_total:.0f}s; overall deadline: {budget}; per-case timeout: {timeout:g}s")
    print(f"Report: {xml_path}")

    print("\n=== CASE RESULTS ===")
    for status, name, reason, output, duration in results:
        suffix = f"  ({reason})" if status != "PASS" else ""
        print(f"[{status}] {duration:8.1f}s  {name}{suffix}")
    print("====================")

    if failures:
        # Echo a diagnosis block at the end of the log so the failing case and
        # its pytest failure section are visible without scrolling the whole run.
        print("\n=== FAILED CASES ===")
        for status, name, reason, output, duration in failures:
            print(f"[{status}] {name}")
            print(f"    Reason: {reason}")
            print(f"    Duration: {duration:.1f}s")
            detail = extract_failure_detail(output) if output else []
            if detail:
                print("    Failure output:")
            elif output:
                print("    No pytest failure section found; last 40 output lines:")
                detail = output.splitlines()[-40:]
            for line in detail:
                print(f"      {line}")
        print("====================")
        print(
            f"Result: FAIL ({counts['FAIL']} failed, {counts['TIMEOUT']} timeout, "
            f"{counts['ERROR']} error, {counts['SKIPPED']} not run)"
        )
        for status, name, reason, output, duration in failures[:10]:
            print(f"::error::FLA UT {status}: {name} ({reason})", file=sys.stderr, flush=True)
        if len(failures) > 10:
            print(
                f"::error::FLA UT: {len(failures) - 10} more failing case(s) listed in the log",
                file=sys.stderr,
                flush=True,
            )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FLA PR tests from a caselist.")
    parser.add_argument("--caselist", required=True, help="Path to the caselist file")
    parser.add_argument(
        "--build-dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
        help="Working directory for test commands (default: repo root)",
    )
    parser.add_argument("--output-xml", default=OUTPUT_XML, help="JUnit XML output path")
    parser.add_argument("--timeout", type=float, default=1200, help="Per-command timeout in seconds")
    parser.add_argument(
        "--deadline",
        type=float,
        default=0,
        help="Overall test budget in seconds; later commands are skipped (0 disables)",
    )
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.deadline < 0:
        parser.error("--deadline must not be negative")
    run_cases(os.path.abspath(args.caselist), os.path.abspath(args.build_dir), args.output_xml, args.timeout, args.deadline)
