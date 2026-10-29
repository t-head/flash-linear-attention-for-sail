#!/usr/bin/env bash
# CI entry point for the GDN performance gate.
#
#   ci_perf_gate.sh <pr-checkout> <target-ref> <artifact-dir> [extra benchmark args...]
#
# The base arm is cut from <target-ref> with `git worktree`, so the job needs a
# checkout that can resolve that ref (shallow clones must fetch it first).
#
# Env knobs:
#   PERF_CONTAINER  GPU container name (default: ppu-bench).
#                   Set it to the empty string when the job already runs INSIDE
#                   the GPU container (e.g. t-head/ci-infra/cluster-run).
#   PERF_DEVICE     device index inside container (default: 0)
#   PERF_HEADS      grid heads  (default: 16,32,64,128 = the CI full tier;
#                   the CI fast tier passes 32)
#   PERF_SEQLENS    grid sequence lengths         (default: 16384)
#   PERF_SWEEPS     full-grid sweeps per arm      (default: 4, first dropped)
#                   The verdict is the median of the rest, so a single disturbed
#                   sweep cannot decide it; three was one blip away from a
#                   false verdict at the 5% band.
#   PERF_TOL        allowed slowdown before FAIL  (default: 0.03)
#   PERF_TOL_CP     same for the CP pre-process units (default: 0.05)
#   PERF_CP         1 = also time the CP pre-process units (default: 1). They sit
#                   behind chunk_gated_delta_rule(cp_context=...), so the e2e grid
#                   never launches them; see --gate-phase below.
#   PERF_GATE_PHASE phases judged by the verdict (default: total,cp_fwd,cp_bwd)
#                   Both the e2e total and the CP pre-process units are judged
#                   for regressions; see PERF_PROFILE_ITERS and PERF_GATE_LEVEL
#                   below.
#   PERF_ANNOTATION annotation level for regressed points: error|warning
#                   (default: error). Advisory runs use warning so the PR page
#                   does not look like a failure when the check stays green.
#   PERF_PROFILE_ITERS  torch.profiler iterations   (default: 5). The verdict
#                   reads per-kernel timings to judge each kernel on its own
#                   time, which is invisible in the e2e total. 0 turns that
#                   off, and the gate then refuses to judge kernels (exit 3)
#                   unless PERF_GATE_LEVEL=phase is set here as well. Profiling
#                   costs 3 warmup + N iterations per phase per point per arm,
#                   so keep N small on the full tier.
#   PERF_GATE_LEVEL     kernel|phase, what the regression check judges
#                   (default: kernel = every kernel on its own time)
#   PERF_KERNEL_TOL     a kernel may get this much slower (relative to itself)
#                   before the run falls back to PERF_KERNEL_TOL_ABS: the
#                   allowance is max(rel * base, abs), so either bound passing
#                   is enough. The two meet at abs / rel, i.e. at 0.400 ms for
#                   the defaults -- shorter kernels are judged on the floor,
#                   longer ones on the percentage (default: 0.02)
#   PERF_KERNEL_TOL_ABS absolute floor for that bound in ms, so a short kernel
#                   is not failed on launch jitter. It is sized on measurement:
#                   in the full grid the untouched 95 us recompute_w_u kernel
#                   read 6 us (6.3%) apart between the arms, while its 370 us
#                   reading read 0.4 us apart. A bare 2% bound would fail the
#                   short reading on noise, so the floor is 8 us (default: 0.008)
#   BENCH_SCRIPT    path to benchmark_gdn_all.py  (default: sibling file)
#   PREFLIGHT_ONLY  1 = resolve the base arm and exit
#
# Exit: 0 pass | 2 regression | 3 infra/config | 1 wrapper misuse
set -u

REPO=${1:?usage: ci_perf_gate.sh <pr-checkout> <target-ref> <artifact-dir>}
TARGET=${2:?usage: ci_perf_gate.sh <pr-checkout> <target-ref> <artifact-dir>}
OUT=${3:?usage: ci_perf_gate.sh <pr-checkout> <target-ref> <artifact-dir>}
shift 3

HERE=$(cd "$(dirname "$0")" && pwd)
BENCH=${BENCH_SCRIPT:-$HERE/benchmark_gdn_all.py}
CONTAINER=${PERF_CONTAINER-ppu-bench}
DEVICE=${PERF_DEVICE:-0}
HEADS=${PERF_HEADS:-16,32,64,128}
SEQLENS=${PERF_SEQLENS:-16384}
SWEEPS=${PERF_SWEEPS:-4}
TOL=${PERF_TOL:-0.03}
TOL_CP=${PERF_TOL_CP:-0.05}
CP=${PERF_CP:-1}
GATE_PHASE=${PERF_GATE_PHASE:-total,cp_fwd,cp_bwd}
ANNOTATION=${PERF_ANNOTATION:-error}
PROFILE_ITERS=${PERF_PROFILE_ITERS:-5}
KERNEL_TOL=${PERF_KERNEL_TOL:-0.02}
KERNEL_TOL_ABS=${PERF_KERNEL_TOL_ABS:-0.008}
GATE_LEVEL=${PERF_GATE_LEVEL:-kernel}

command -v python3 >/dev/null || { echo "[ci] python3 not found" >&2; exit 3; }
[ -f "$BENCH" ] || { echo "[ci] benchmark script not found: $BENCH" >&2; exit 3; }
[ -d "$REPO/.git" ] || { echo "[ci] $REPO is not a git checkout; use --base-archive instead" >&2; exit 3; }

mkdir -p "$OUT" || exit 3
# actions/checkout mounts the workspace with a different owner inside job
# containers; without this `git worktree add` fails on dubious ownership
git config --global --add safe.directory "$REPO" 2>/dev/null || true

# one job per artifact dir; a held lock means a previous run is still going
if command -v flock >/dev/null; then
  exec 9>"$OUT/.lock" || exit 3
  flock -w 7200 9 || { echo "[ci] another perf job holds $OUT/.lock" >&2; exit 3; }
fi

COMMON=(--opt-repo "$REPO" --base-mode git --base-ref "$TARGET"
        --device "$DEVICE" --heads "$HEADS" --seq-lens "$SEQLENS"
        --profile-iters "$PROFILE_ITERS" --out-dir "$OUT")
[ -n "$CONTAINER" ] && COMMON+=(--docker-container "$CONTAINER")

if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
  python3 "$BENCH" "${COMMON[@]}" --provision-only
  exit $?
fi

echo "[ci] base=$TARGET opt=$REPO heads=$HEADS T=$SEQLENS sweeps=$SWEEPS tol=$TOL cp_tol=$TOL_CP level=$GATE_LEVEL"
# The verdict judges the e2e phases and the CP pre-process units for
# regressions, and at kernel level every kernel's own time as well. The CP units
# only run under cp_context, so they are timed through their wrappers in the
# worker (all_gather stubbed, one rank) and their kernels are profiled there. The
# e2e kernel rows come from the module profiling pass. The single-kernel launch
# units (--kernels) stay a manual tool and are not judged here.
GATE_ARGS=(--gate --gate-phase "$GATE_PHASE" --gate-tol "$TOL" --gate-tol-cp "$TOL_CP"
           --gate-gain-level "$GATE_LEVEL"
           --gate-kernel-tol "$KERNEL_TOL" --gate-kernel-tol-abs "$KERNEL_TOL_ABS")
[ "$CP" = "1" ] || GATE_ARGS+=(--no-cp)
set +e
python3 "$BENCH" "${COMMON[@]}" --sweeps "$SWEEPS" --drop-sweeps 1 "${GATE_ARGS[@]}" "$@"
rc=$?
set -e

echo "----- gate.md -----"
[ -f "$OUT/gate.md" ] && cat "$OUT/gate.md"
# fail each regressed point on the PR page (GitHub annotations; plain text elsewhere).
# python instead of jq: the actions-runner image has no jq, and a silent miss here
# would lose every annotation while the job still exits correctly.
if [ -f "$OUT/gate.json" ]; then
  python3 - "$OUT/gate.json" "$ANNOTATION" <<'PY' || true
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(0)
for c in d.get("failing") or []:
    print(f"::{sys.argv[2]}::{c.get('reason','fail')} H={c.get('H')} T={c.get('T')} "
          f"{c.get('phase')} {c.get('speedup', 0):.3f}x "
          f"({c.get('base_ms', 0):.3f} -> {c.get('opt_ms', 0):.3f} ms)")
# A kernel regression is what a blocking verdict rejects, so it belongs on the
# PR page next to the phase ones instead of only in the job log.
for r in (d.get("kernel_gain") or {}).get("regressions") or []:
    print(f"::{sys.argv[2]}::kernel regression {r.get('kernel')} {r.get('phase')} "
          f"H={r.get('H')} T={r.get('T')} +{r.get('delta_ms', 0):.3f}ms > "
          f"{r.get('tol_ms', 0):.3f}ms ({r.get('rule', '')})")
PY
fi
echo "[ci] exit=$rc (0 pass / 2 regression / 3 infra)"
exit $rc
