#!/usr/bin/env bash
# Run the two-trial experiment on the remote GPU server.
# Usage (from your machine, in the project root):
#   chmod +x run_trials_remote.sh
#   ./run_trials_remote.sh
#
# Or on the server after cloning/copying the repo:
#   python run_trials.py

set -e
REMOTE="ubuntu@216.81.248.162"
KEY="${1:-private_key.pem}"
# On the server, project might be e.g. ~/stat4830project or ~/prompt-optimization
PROJECT_DIR="${2:-stat4830project}"

if [[ "$REMOTE_RUN" == "1" ]]; then
  # Already on remote (e.g. ssh ... "REMOTE_RUN=1 bash -s" < run_trials_remote.sh)
  cd "$PROJECT_DIR" && python run_trials.py
  exit 0
fi

if [[ ! -f "$KEY" ]]; then
  echo "Usage: $0 [private_key.pem] [remote_project_dir]"
  echo "  private_key.pem should be next to this script (or pass path)."
  echo "  Or run on the server: python run_trials.py"
  exit 1
fi

echo "Running trials on $REMOTE (GEPA off then GEPA on, 150 steps or 5 min each)..."
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$REMOTE" "cd $PROJECT_DIR && python run_trials.py"
