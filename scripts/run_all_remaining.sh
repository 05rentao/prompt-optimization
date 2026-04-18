#!/usr/bin/env bash
# Master runner for the remaining experiments on feat/ablation-coevolution.
#
# Phases (in order):
#   1. R15 HarmBench baseline (no adversary, no defense)                 ~30 min
#   2. R15 XSTest target-only eval (paired with the baseline)            ~30-60 min
#   3. R11 adversary RLOO + length penalty re-run, saving adapters       ~5 h
#   4. XSTest eval on the R11 checkpoint (adversary mode)                <1 h
#   5. R14 staged co-evolution with RLOO, saving final adapters          ~10 h
#   6. XSTest eval on the R14 checkpoint (adversary mode)                <1 h
#
# NOTE: configs/xstest_batch_eval.yaml runs the same target-only XSTest pass
# as phase 2, so a separate "xstest target-only baseline" phase would just
# duplicate phase 2's output in a different directory. It's retained as a
# standalone smoke-test config (see SMOKE TEST block above) but is not part
# of the full pipeline.
#
# SMOKE TEST (run first, before the full pipeline):
#   PROMPT_OPT_CONFIG_PATH=configs/xstest_batch_eval.yaml MODE=xstest bash scripts/launch_unified_prime.sh
# Expected: ~30-60 min, ~$2. If this works, the full pipeline infrastructure is solid.
# (Same invocation as phase 5 but in isolation — useful for confirming HF_TOKEN,
# vLLM startup, and the XSTest loader all work before burning ~$36 of GPU time.)
#
# Design:
#   - Each phase writes stdout+stderr to logs/run_all_remaining/phase<N>_<ts>.log
#     (appended also to the shared phase console) so nothing is lost.
#   - `set -euo pipefail` aborts the whole script on the first failure.
#   - Every phase uses the same reflection vLLM server spun up by the
#     per-phase invocation of scripts/launch_unified_prime.sh. We do NOT try
#     to keep a single vLLM up across phases here — the launcher starts a
#     fresh server each time so config-driven model changes just work. If you
#     want a shared server, export KEEP_VLLM_UP=1 and start vLLM by hand.
#   - Summary at the end points at the per-phase artifact directories.
#
# Usage:
#   bash scripts/run_all_remaining.sh
# Optional environment:
#   SKIP_PHASES="3 4"   # skip specific phases by number

set -euo pipefail

# HF_TOKEN must be present before anything starts — the XSTest loader and
# gated HF model downloads both require it. Fail fast here so we do not spin
# up vLLM and then blow up hours in.
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set. Export it before running (needed for the XSTest dataset"
  echo "       loader at src/dataset/xstest_loader.py and for gated HF models)."
  echo ""
  echo "       Example:"
  echo "         export HF_TOKEN=hf_xxx..."
  echo "         bash scripts/run_all_remaining.sh"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs/run_all_remaining/${TIMESTAMP}"
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

# Run a phase, streaming output to both the console and a timestamped log file.
# If the phase exits non-zero, print a banner and exit so GPU time stops being
# burned on downstream phases that will blow up anyway.
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

print_summary() {
  echo ""
  echo "================================================================"
  echo "=== Summary"
  echo "================================================================"
  for line in "${PHASE_SUMMARIES[@]}"; do
    echo "  ${line}"
  done
  echo ""
  echo "Artifact roots:"
  echo "  results/r15_baseline/              (phases 1-2, HarmBench + XSTest baseline)"
  echo "  results/r11_rloo_length_penalty/   (phase 3, adapters under checkpoints/)"
  echo "  results/r11_xstest_eval/           (phase 4)"
  echo "  results/r14_coev_rloo/             (phase 5, adapters under checkpoints/)"
  echo "  results/r14_xstest_eval/           (phase 6)"
  echo ""
  echo "Logs: ${LOG_DIR}"
}

# Phase 1 — R15 HarmBench baseline (~30 min, ~$1)
# Cheap control: no adversary, no defense, pure target + judge. Runs first so
# every downstream phase has a comparable baseline number regardless of which
# later phase(s) fail.
run_phase 1 "R15 HarmBench baseline (no adversary, no defense)" "r15_baseline" \
  env PROMPT_OPT_CONFIG_PATH="configs/r15_baseline.yaml" MODE="baseline" \
  bash scripts/launch_unified_prime.sh

# Phase 2 — R15 XSTest target-only (~30-60 min, ~$2)
# Over-refusal companion for the baseline; results land in results/r15_baseline/.
run_phase 2 "R15 XSTest target-only (paired with baseline)" "r15_xstest" \
  env PROMPT_OPT_CONFIG_PATH="configs/r15_xstest_eval.yaml" MODE="xstest" \
  bash scripts/launch_unified_prime.sh

# Phase 3 — R11 retrain (~5h, ~$10 on H100)
run_phase 3 "R11 adversary RLOO + length penalty (retrain with checkpointing)" "r11_train" \
  env PROMPT_OPT_CONFIG_PATH="configs/r11_rloo_length_penalty.yaml" MODE="adversary" \
  bash scripts/launch_unified_prime.sh

# Phase 4 — XSTest on R11 (<1h, ~$2)
run_phase 4 "XSTest eval on R11 checkpoint" "r11_xstest" \
  env PROMPT_OPT_CONFIG_PATH="configs/r11_xstest_eval.yaml" MODE="xstest" \
  bash scripts/launch_unified_prime.sh

# Phase 5 — R14 coev RLOO (~10h, ~$20)
run_phase 5 "R14 co-evolution with RLOO" "r14_train" \
  env PROMPT_OPT_CONFIG_PATH="configs/r14_coev_rloo.yaml" MODE="coev_v2_rloo" \
  bash scripts/launch_unified_prime.sh

# Phase 6 — XSTest on R14 (<1h, ~$2)
run_phase 6 "XSTest eval on R14 checkpoint" "r14_xstest" \
  env PROMPT_OPT_CONFIG_PATH="configs/r14_xstest_eval.yaml" MODE="xstest" \
  bash scripts/launch_unified_prime.sh

echo ""
echo "=== All experiments complete ==="
print_summary
