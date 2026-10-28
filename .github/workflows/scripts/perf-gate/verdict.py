# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Judge a stored GDN gate result under one policy.

Two steps must reach the same verdict: a fresh campaign, and the replay of the
stored result for a head whose campaign is not rerun (a description edit, an
unrelated label, a review event). The replay used to look only at
``verdict == "fail"``, which missed every rule the workflow applied on top of
it, so a head that failed one of them came back green.

The gate script folds all of them into ``verdict`` (the phase tolerances and,
at kernel level, each kernel's own allowance), so the stored file is the
single source of truth and this script only has to enforce it.

Usage:
  verdict.py <gate.json> <block:true|false>

Exit: 0 pass, 2 fail, 3 infra. ``infra`` means nothing was measured: that is
not a pass, so a blocking gate fails on it. Returning 0 there made a campaign
that died on the pod timeout (or on one bad point) report a green check, which
is the one answer a gate must never give.
"""
import json
import statistics
import sys


def error(message, code=2):
    print("::error::" + message)
    sys.exit(code)


def main():
    path = sys.argv[1]
    blocking = sys.argv[2].strip()
    with open(path) as fh:
        d = json.load(fh)
    verdict = d.get("verdict")
    pooled = d.get("pooled") or {}
    cells = d.get("cells") or []
    kg = d.get("kernel_gain") or {}
    # Half the sweep range, per point: how much this board jitters on this
    # grid. Printed so a thin margin can be told apart from noise.
    noise = statistics.median([(c["max"] - c["min"]) / 2 for c in cells]) if cells else 0.0
    pooled_text = ", ".join(f"{ph} {v:.3f}x" for ph, v in pooled.items())
    print(f"verdict={verdict} pooled={pooled_text} | sweep spread +/-{noise * 100:.2f}%")

    # An advisory board reports only: even an infra run is not its business.
    if blocking != "true":
        print("policy is advisory on this board; reporting only")
        return 0

    # Nothing was measured, so no kernel was shown to hold its time. Name the
    # first problems -- they say whether the pod never started or one point
    # broke -- and exit 3, the same code the gate uses for infrastructure.
    if verdict == "infra":
        for problem in (d.get("problems") or [])[:4]:
            print("::error::infra: " + problem)
        error("this result measured nothing (infra); a blocking gate cannot pass "
              "on a campaign that did not run", 3)

    slower = kg.get("regressions") or []
    # A regression and an infra problem both mean "no evidence the PR holds
    # its speed", which is what a declared change has to give. Name the slower
    # kernels first: the summary line alone reads as "something failed", which
    # is not actionable when the cause is one kernel.
    if verdict != "pass":
        if slower:
            detail = ", ".join(
                f"{r.get('kernel')} {r.get('phase')} H={r.get('H')} T={r.get('T')} "
                f"+{r.get('delta_ms', 0.0):.3f}ms > {r.get('tol_ms', 0.0):.3f}ms "
                f"({r.get('rule', '')})"
                for r in slower[:6])
            print("::error::kernel(s) slower than the tolerance allows: " + detail)
        error("perf gate did not pass; the failing points are in gate.md")

    best = kg.get("best") or {}
    if best:
        print(f"fastest kernel {best.get('kernel')} ({best.get('phase')}) "
              f"{best.get('speedup', 0.0):.3f}x over "
              f"{len(kg.get('units') or [])} kernel(s), {len(slower)} kernel "
              "regression(s); no gain was required")
    return 0


if __name__ == "__main__":
    sys.exit(main())
