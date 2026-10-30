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

Exit: 0 pass, 2 fail, 3 infra. ``infra`` means the campaign left no usable
answer: either nothing was measured, or a point breached by less than its own
sweep scatter and cannot be resolved at all. That is not a pass -- a blocking
gate exists to be told the code holds its speed, and an unreadable measurement
does not tell it. Re-run to measure again.
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
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError) as e:
        # Copying the verdict to the NAS is the last thing the pod does, so a
        # kill at the timeout can leave that copy half written, and a replay can
        # be pointed at a file an older campaign never finished. Either way no
        # verdict was left to judge: say so under the code a blocking gate
        # already understands, instead of a traceback where a verdict belongs.
        error(f"{path} cannot be read ({type(e).__name__}: {e}); no verdict was "
              f"published to judge -- re-run to measure again", 3)
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

    # Nothing usable was measured, so no kernel was shown to hold its time.
    # Name the problems first -- they say whether the pod never started, or a
    # point breached inside its own sweep scatter -- and exit 3, the same code
    # the gate uses for infrastructure.
    if verdict == "infra":
        for problem in (d.get("problems") or [])[:4]:
            print("::error::infra: " + problem)
        # Say whether the campaign already spent another round on this point, so
        # "re-run" is not advice the author has already paid for.
        again = (d.get("sweeps") or {}).get("remeasured") or 0
        remedy = (f"re-measured {again} round(s) and still unreadable" if again
                  else "re-run to measure again")
        error("this result is not a measurement (infra); a blocking gate cannot "
              f"pass on it -- {remedy}", 3)

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
