#!/usr/bin/env bash
# Standalone R12 runner — RLOO + length penalty + KL penalty.
#
# NOT wired into scripts/run_all_remaining.sh: we want to review R11 numbers
# first and decide whether R12 beats R11 before lining it up in the main
# pipeline (and before deciding which config feeds R14's co-evolution loop).
#
# Phases:
#   1. R12 adversary training (120 iters, RLOO, length+KL reward shaping)
#   2. XSTest eval on R12 checkpoint (adversary mode)
#
# Matches scripts/run_all_remaining.sh conventions:
#   - `set -euo pipefail` + fail-fast on any phase error
#   - Each phase writes stdout+stderr to logs/r12/<ts>/phaseN_<name>.log
#   - Reuses scripts/launch_unified_prime.sh for vLLM startup + config dispatch
#
# Usage:
#   bash scripts/run_r12.sh
# Optional env:
#   SKIP_PHASES="2"            # skip XSTest eval phase
#   KEEP_VLLM_UP=1             # persist the reflection vLLM across phases (not
#                              # automatic — launch_unified_prime.sh spins a new
#                              # server per invocation unless you override it)

set -euo pipefail

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set. Export it before running (needed for XSTest"
  echo "       dataset loader and gated HF models)."
  echo ""
  echo "       export HF_TOKEN=hf_xxx..."
  echo "       bash scripts/run_r12.sh"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs/r12/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

SKIP_PHASES="${SKIP_PHASES:-}"

declare -a PHASE_SUMMARIES=()

should_skip() {
  local phase_num="$1"
  for skipped in ${SKIP_PHASES}; do
    if [[ "${skipped}" == "${phase_num}" ]]; then
      return 0
    fi
  done
  return 1
}

print_summary() {
  echo ""
  echo "================================================================"
  echo "=== R12 Summary"
  echo "================================================================"
  for line in "${PHASE_SUMMARIES[@]}"; do
    echo "  ${line}"
  done
  echo ""
  echo "Artifact roots:"
  echo "  results/r12_rloo_kl_penalty/       (phase 1, adapters under checkpoints/)"
  echo "  results/r12_xstest_eval/           (phase 2)"
  echo ""
  echo "Logs: ${LOG_DIR}"
}

run_phase() {
  local phase_num="$1"
  local phase_name="$2"
  local log_name="$3"
  shift 3

  if should_skip "${phase_num}"; then
    echo "=== Phase ${phase_num}: ${phase_name} — SKIPPED via SKIP_PHASES ==="
    PHASE_SUMMARIES+=("${phase_num}. ${phase_name} — skipped")
    return
  fi

  local log_file="${LOG_DIR}/phase${phase_num}_${log_name}.log"
  echo ""
  echo "================================================================"
  echo "=== Phase ${phase_num}: ${phase_name}"
  echo "=== Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "=== Log:     ${log_file}"
  echo "================================================================"

  set +e
  ("$@") 2>&1 | tee "${log_file}"
  local rc=${PIPESTATUS[0]}
  set -e

  if [[ ${rc} -ne 0 ]]; then
    echo ""
    echo "!!! Phase ${phase_num} (${phase_name}) FAILED with exit code ${rc}."
    echo "!!! See: ${log_file}"
    echo "!!! Aborting before downstream phases burn more GPU time."
    PHASE_SUMMARIES+=("${phase_num}. ${phase_name} — FAILED (${log_file})")
    print_summary
    exit ${rc}
  fi

  echo ""
  echo "=== Phase ${phase_num} complete: ${phase_name}"
  PHASE_SUMMARIES+=("${phase_num}. ${phase_name} — ok (${log_file})")
}

# Phase 1 — R12 adversary training (~5h on H100, same budget as R11 retrain)
run_phase 1 "R12 adversary RLOO + length penalty + KL penalty" "r12_train" \
  env PROMPT_OPT_CONFIG_PATH="configs/r12_rloo_kl_penalty.yaml" MODE="adversary" \
  bash scripts/launch_unified_prime.sh

# Phase 2 — XSTest eval on the R12 checkpoint (<1h on H100)
run_phase 2 "XSTest eval on R12 checkpoint" "r12_xstest" \
  env PROMPT_OPT_CONFIG_PATH="configs/r12_xstest_eval.yaml" MODE="xstest" \
  bash scripts/launch_unified_prime.sh

echo ""
echo "=== R12 complete ==="
print_summary
