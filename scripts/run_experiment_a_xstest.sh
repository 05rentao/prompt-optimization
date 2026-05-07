#!/usr/bin/env bash
# Run Experiment A: XSTest harmless-prompt / over-refusal comparison.
#
# Conditions:
#   1. Default target-only
#   2. Default target after adversary rewrite
#   3. GEPA-defense target-only
#   4. GEPA-defense target after adversary rewrite (optional; requires R14 KL
#      checkpoints_best unless RUN_ADV_GEPA=0)
#
# Usage:
#   bash scripts/run_experiment_a_xstest.sh
#
# Optional:
#   SKIP_PHASES="1 2" bash scripts/run_experiment_a_xstest.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]]; then
  echo "ERROR: HF_TOKEN or HUGGINGFACE_HUB_TOKEN must be set for Paul/XSTest and gated models."
  exit 1
fi

should_skip() {
  local phase="$1"
  [[ " ${SKIP_PHASES:-} " == *" ${phase} "* ]]
}

run_phase() {
  local phase="$1"
  local label="$2"
  local config="$3"

  if should_skip "${phase}"; then
    echo ""
    echo "=== Skipping phase ${phase}: ${label} ==="
    return
  fi

  echo ""
  echo "=== Phase ${phase}: ${label} ==="
  echo "Config: ${config}"
  env PROMPT_OPT_CONFIG_PATH="${config}" MODE="xstest" bash scripts/launch_unified_prime.sh
}

ensure_seed123_checkpoint_alias() {
  local expected="results/r11_full_prompt_seed123/checkpoints_best"
  local available="seed123_checkpoints_best"

  if [[ -e "${expected}/adapter_model.safetensors" ]]; then
    return
  fi

  if [[ -e "${available}/adapter_model.safetensors" ]]; then
    echo "Creating checkpoint alias: ${expected} -> ${available}"
    mkdir -p "$(dirname "${expected}")"
    ln -sfn "../../${available}" "${expected}"
  fi
}

ensure_seed123_checkpoint_alias

run_phase 1 "Default target-only XSTest" \
  "configs/xstest_batch_eval.yaml"

run_phase 2 "Default target + seed123 adversary rewrite XSTest" \
  "configs/r11_seed123_xstest.yaml"

run_phase 3 "GEPA defense target-only XSTest" \
  "configs/xstest_gepa_defense_target_only.yaml"

if [[ "${RUN_ADV_GEPA:-1}" == "1" ]]; then
  if [[ -e "results/r14_coev_full_prompt_kl/checkpoints_best/adapter_model.safetensors" ]]; then
    run_phase 4 "GEPA defense + R14 KL adversary rewrite XSTest" \
      "configs/r14_coev_full_prompt_kl_xstest_gepa_defense.yaml"
  else
    echo ""
    echo "=== Skipping phase 4: GEPA defense + adversary rewrite ==="
    echo "Missing results/r14_coev_full_prompt_kl/checkpoints_best/adapter_model.safetensors"
    echo "Run/copy the R14 KL checkpoint first, or set RUN_ADV_GEPA=0 to suppress this message."
  fi
fi

echo ""
echo "=== Experiment A XSTest matrix complete ==="
echo "Results:"
echo "  results/xstest_batch_baseline/"
echo "  results/r11_seed123_xstest/"
echo "  results/xstest_gepa_defense_target_only/"
echo "  results/r14_coev_full_prompt_kl_xstest_gepa_defense/"
