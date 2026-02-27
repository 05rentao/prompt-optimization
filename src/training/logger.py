import csv
import os
from datetime import datetime
from typing import Dict, Any

class ExperimentLogger:
    """Logs training metrics and generated text to CSV for analysis."""
    def __init__(self, log_dir: str = "outputs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Unique filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"experiment_{timestamp}.csv")
        
        # Initialize CSV with headers (system_prompt + use_gepa_defender for success-rate plots)
        self.headers = [
            "epoch", "behavior", "adversary_attack",
            "target_response", "reward", "is_jailbroken", "explanation",
            "system_prompt", "use_gepa_defender"
        ]
        with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()

    def log_step(self, data: Dict[str, Any]):
        with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(data)


def log_trial_summary(
    trial_name: str,
    jailbreak_fraction: float,
    steps: int,
    elapsed_seconds: float,
    log_dir: str = "outputs",
) -> str:
    """Append a trial summary (jailbreak fraction, steps, time) to outputs/trial_summaries.txt. Returns path."""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "trial_summaries.txt")
    line = (
        f"{datetime.now().isoformat()} | {trial_name} | "
        f"jailbreak_fraction={jailbreak_fraction:.4f} | steps={steps} | elapsed_sec={elapsed_seconds:.1f}\n"
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
    return path