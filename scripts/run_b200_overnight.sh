#!/usr/bin/env bash
# B200 overnight runner — chain the remaining ablation experiments on one
# machine. Assumes HF weights are cached (HF_TOKEN set) and the vLLM stack
# starts cleanly per invocation of scripts/launch_unified_prime.sh.
#
# Phases (in order):
#   1. R12 full-prompt + KL training        (~5h on B200; configs/r12_full_prompt_kl.yaml)
#   2. XSTest on R12 full-prompt + KL best-checkpoint        (~1h)
#   3. XSTest on R11 full-prompt seed=123 best-checkpoint    (~1h)
#   4. R14 full-prompt co-evolution (RLOO)  (~10h; configs/r14_coev_full_prompt.yaml)
#   5. XSTest on R14 full-prompt best-checkpoint             (~1h)
#   6. R14 full-prompt + KL co-evolution (RLOO + kl_coeff=0.05)  (~10h)
#   7. XSTest on R14 full-prompt + KL best-checkpoint        (~1h)
#
# Phase 3 assumes results/r11_full_prompt_seed123/checkpoints_best/ already
# exists (produced by an earlier run of configs/r11_full_prompt_seed123.yaml).
# If the directory is missing, phase 3 will fail fast and abort phases 4-5.
#
# Matches scripts/run_all_remaining.sh conventions:
#   - `set -euo pipefail` + fail-fast on any phase error
#   - Each phase writes stdout+stderr to logs/b200_overnight/<ts>/phaseN_<name>.log
#   - HF_TOKEN precheck before any GPU work starts
#
# Usage:
#   bash scripts/run_b200_overnight.sh
# Optional env:
#   SKIP_PHASES="3 5"          # skip specific phases by number
#   KEEP_VLLM_UP=1             # persist reflection vLLM across phases (must be
#                              # honored by launch_unified_prime.sh manually)

set -euo pipefail

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set. Export it before running (needed for the XSTest"
  echo "       dataset loader and gated HF models)."
  echo ""
  echo "       export HF_TOKEN=hf_xxx..."
  echo "       bash scripts/run_b200_overnight.sh"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs/b200_overnight/${TIMESTAMP}"
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
  echo "=== B200 overnight summary"
  echo "================================================================"
  for line in "${PHASE_SUMMARIES[@]}"; do
    echo "  ${line}"
  done
  echo ""
  echo "Artifact roots:"
  echo "  results/r12_full_prompt_kl/              (phase 1, adapters under checkpoints/ + checkpoints_best/)"
  echo "  results/r12_full_prompt_kl_xstest/       (phase 2)"
  echo "  results/r11_seed123_xstest/              (phase 3)"
  echo "  results/r14_coev_full_prompt/            (phase 4, adapters under checkpoints/ + checkpoints_best/)"
  echo "  results/r14_coev_full_prompt_xstest/     (phase 5)"
  echo "  results/r14_coev_full_prompt_kl/         (phase 6, adapters under checkpoints/ + checkpoints_best/)"
  echo "  results/r14_coev_full_prompt_kl_xstest/  (phase 7)"
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

# Phase 1 — R12 full-prompt + KL training (RLOO + length penalty + kl_coeff=0.05)
run_phase 1 "R12 full-prompt + KL training" "r12_full_prompt_kl_train" \
  env PROMPT_OPT_CONFIG_PATH="configs/r12_full_prompt_kl.yaml" MODE="adversary" \
  bash scripts/launch_unified_prime.sh

# Phase 2 — XSTest on R12 KL checkpoints_best/
run_phase 2 "XSTest eval on R12 full-prompt + KL best-checkpoint" "r12_full_prompt_kl_xstest" \
  env PROMPT_OPT_CONFIG_PATH="configs/r12_full_prompt_kl_xstest.yaml" MODE="xstest" \
  bash scripts/launch_unified_prime.sh

# Phase 3 — XSTest on R11 full-prompt seed=123 checkpoints_best/
# Prereq: the seed=123 training run must have produced checkpoints_best/.
run_phase 3 "XSTest eval on R11 full-prompt seed=123 best-checkpoint" "r11_seed123_xstest" \
  env PROMPT_OPT_CONFIG_PATH="configs/r11_seed123_xstest.yaml" MODE="xstest" \
  bash scripts/launch_unified_prime.sh

# Phase 4 — R14 co-evolution with full 2262-char prompt, RLOO adversary
run_phase 4 "R14 full-prompt co-evolution (RLOO)" "r14_coev_full_prompt_train" \
  env PROMPT_OPT_CONFIG_PATH="configs/r14_coev_full_prompt.yaml" MODE="coev_v2_rloo" \
  bash scripts/launch_unified_prime.sh

# Phase 5 — XSTest on R14 checkpoints_best/
run_phase 5 "XSTest eval on R14 full-prompt best-checkpoint" "r14_coev_full_prompt_xstest" \
  env PROMPT_OPT_CONFIG_PATH="configs/r14_coev_full_prompt_xstest.yaml" MODE="xstest" \
  bash scripts/launch_unified_prime.sh

# Phase 6 — R14 co-evolution with the R12-equivalent KL penalty layered on top
# of the RLOO + length-penalty recipe used in phase 4. Single-knob ablation
# against results/r14_coev_full_prompt/.
run_phase 6 "R14 full-prompt + KL co-evolution (RLOO)" "r14_coev_full_prompt_kl_train" \
  env PROMPT_OPT_CONFIG_PATH="configs/r14_coev_full_prompt_kl.yaml" MODE="coev_v2_rloo" \
  bash scripts/launch_unified_prime.sh

# Phase 7 — XSTest on the R14 + KL best-checkpoint (stage-boundary peak ASR).
run_phase 7 "XSTest eval on R14 full-prompt + KL best-checkpoint" "r14_coev_full_prompt_kl_xstest" \
  env PROMPT_OPT_CONFIG_PATH="configs/r14_coev_full_prompt_kl_xstest.yaml" MODE="xstest" \
  bash scripts/launch_unified_prime.sh

echo ""
echo "=== B200 overnight complete ==="
print_summary
