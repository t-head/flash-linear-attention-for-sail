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
#                   the GPU container (e.g. t-head/ppu-scheduler-action).
#   PERF_DEVICE     device index inside container (default: 0)
#   PERF_HEADS      grid heads                    (default: 32,64)
#   PERF_SEQLENS    grid sequence lengths         (default: 8192,16384)
#   PERF_SWEEPS     full-grid sweeps per arm      (default: 3)
#   PERF_TOL        allowed slowdown before FAIL  (default: 0.03)
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
HEADS=${PERF_HEADS:-32,64}
SEQLENS=${PERF_SEQLENS:-8192,16384}
SWEEPS=${PERF_SWEEPS:-3}
TOL=${PERF_TOL:-0.03}

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
        --profile-iters 2 --out-dir "$OUT")
[ -n "$CONTAINER" ] && COMMON+=(--docker-container "$CONTAINER")

if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
  python3 "$BENCH" "${COMMON[@]}" --provision-only
  exit $?
fi

echo "[ci] base=$TARGET opt=$REPO heads=$HEADS T=$SEQLENS sweeps=$SWEEPS tol=$TOL"
set +e
python3 "$BENCH" "${COMMON[@]}" --sweeps "$SWEEPS" --drop-sweeps 1 \
  --gate --gate-phase total --gate-tol "$TOL" "$@"
rc=$?
set -e

echo "----- gate.md -----"
[ -f "$OUT/gate.md" ] && cat "$OUT/gate.md"
# fail each regressed point on the PR page (GitHub annotations; plain text elsewhere).
# python instead of jq: the actions-runner image has no jq, and a silent miss here
# would lose every annotation while the job still exits correctly.
if [ -f "$OUT/gate.json" ]; then
  python3 - "$OUT/gate.json" <<'PY' || true
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(0)
for c in d.get("failing") or []:
    print(f"::error::{c.get('reason','fail')} H={c.get('H')} T={c.get('T')} "
          f"{c.get('phase')} {c.get('speedup', 0):.3f}x "
          f"({c.get('base_ms', 0):.3f} -> {c.get('opt_ms', 0):.3f} ms)")
PY
fi
echo "[ci] exit=$rc (0 pass / 2 regression / 3 infra)"
exit $rc
