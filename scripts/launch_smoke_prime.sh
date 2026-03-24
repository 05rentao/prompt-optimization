#!/usr/bin/env bash
set -euo pipefail

# Launch minimal-budget smoke tests. Config from configs/smoke.yaml or smoke_eval.yaml.
#
# Examples:
#   bash scripts/launch_smoke_prime.sh
#   RUN_KIND=eval bash scripts/launch_smoke_prime.sh
#   MODE=coev_v2_rloo bash scripts/launch_smoke_prime.sh
#
# Edit scripts.unified_runner in the YAML for coev_v2_mode, run_kind, etc.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${MODE:-all}"            # all | gepa | coev_v2 | coev_v2_rloo | adversary
RUN_KIND="${RUN_KIND:-train}"  # train | eval — selects smoke.yaml vs smoke_eval.yaml
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

if [[ -n "${PROMPT_OPT_CONFIG_PATH:-}" ]]; then
  :
elif [[ "${RUN_KIND}" == "eval" ]]; then
  export PROMPT_OPT_CONFIG_PATH="${ROOT_DIR}/configs/smoke_eval.yaml"
else
  export PROMPT_OPT_CONFIG_PATH="${SMOKE_CONFIG_PATH:-${ROOT_DIR}/configs/smoke.yaml}"
fi

echo "Using smoke config: ${PROMPT_OPT_CONFIG_PATH}"
if [[ ! -f "${PROMPT_OPT_CONFIG_PATH}" ]]; then
  echo "ERROR: config not found at ${PROMPT_OPT_CONFIG_PATH}"
  exit 1
fi

run_one() {
  local run_mode="$1"

  case "${run_mode}" in
    gepa)
      MODE=gepa bash scripts/launch_unified_prime.sh
      ;;
    coev_v2)
      MODE=coev_v2 bash scripts/launch_unified_prime.sh
      ;;
    coev_v2_rloo)
      MODE=coev_v2_rloo bash scripts/launch_unified_prime.sh
      ;;
    adversary)
      MODE=adversary bash scripts/launch_unified_prime.sh
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
    run_one coev_v2
    run_one coev_v2_rloo
    run_one adversary
    ;;
  gepa|coev_v2|coev_v2_rloo|adversary)
    run_one "${MODE}"
    ;;
  *)
    echo "ERROR: MODE must be one of: all, gepa, coev_v2, coev_v2_rloo, adversary"
    exit 1
    ;;
esac

echo "Smoke run complete."
echo "Artifacts: scripts.unified_runner paths in ${PROMPT_OPT_CONFIG_PATH}. RUN_TAG=${RUN_TAG} is for logging only."
