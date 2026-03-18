#!/usr/bin/env bash
set -euo pipefail

# Launch minimal-budget smoke tests without editing configs/default.yaml.
# Supports one mode or all four sequentially.
#
# Examples:
#   bash scripts/launch_smoke_prime.sh
#   MODE=all RUN_KIND=eval bash scripts/launch_smoke_prime.sh
#   MODE=coev COEV_MODE=gepa bash scripts/launch_smoke_prime.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${MODE:-all}"            # all | gepa | coev | coev_v2 | adversary
RUN_KIND="${RUN_KIND:-train}"  # train | eval
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

SMOKE_CONFIG_PATH="${SMOKE_CONFIG_PATH:-${ROOT_DIR}/configs/smoke.yaml}"
export PROMPT_OPT_CONFIG_PATH="${SMOKE_CONFIG_PATH}"

echo "Using smoke config: ${PROMPT_OPT_CONFIG_PATH}"
if [[ ! -f "${PROMPT_OPT_CONFIG_PATH}" ]]; then
  echo "ERROR: smoke config not found at ${PROMPT_OPT_CONFIG_PATH}"
  exit 1
fi

# Keep explicit envs for consistency with smoke defaults and easy overrides.
export TRAIN_SIZE="${TRAIN_SIZE:-8}"
export VAL_SIZE="${VAL_SIZE:-4}"
export MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-8}"
export MAX_TOKENS="${MAX_TOKENS:-96}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export EVAL_METHOD="${EVAL_METHOD:-heuristic}"
export COEV_MODE="${COEV_MODE:-reinforce}"
export COEV_V2_MODE="${COEV_V2_MODE:-coev}"
export ADVERSARY_MODE="${ADVERSARY_MODE:-train}"

if [[ "${RUN_KIND}" == "eval" ]]; then
  export COEV_MODE="eval"
  export COEV_V2_MODE="eval"
  export ADVERSARY_MODE="eval"
fi

run_one() {
  local run_mode="$1"
  local results_root="results/smoke/${RUN_TAG}"
  mkdir -p "${results_root}"

  case "${run_mode}" in
    gepa)
      GEPA_RESULTS_DIR="${results_root}/gepa" MODE=gepa bash scripts/launch_unified_prime.sh
      ;;
    coev)
      COEV_RESULTS_DIR="${results_root}/coev" MODE=coev bash scripts/launch_unified_prime.sh
      ;;
    coev_v2)
      COEV_RESULTS_DIR="${results_root}/coev_v2" MODE=coev_v2 bash scripts/launch_unified_prime.sh
      ;;
    adversary)
      ADVERSARY_RESULTS_DIR="${results_root}/adversary" MODE=adversary bash scripts/launch_unified_prime.sh
      ;;
    *)
      echo "ERROR: unsupported mode '${run_mode}'"
      exit 1
      ;;
  esac
}

case "${MODE}" in
  all)
    run_one gepa
    run_one coev
    run_one coev_v2
    run_one adversary
    ;;
  gepa|coev|coev_v2|adversary)
    run_one "${MODE}"
    ;;
  *)
    echo "ERROR: MODE must be one of: all, gepa, coev, coev_v2, adversary"
    exit 1
    ;;
esac

echo "Smoke run complete."
echo "Artifacts root: results/smoke/${RUN_TAG}"
