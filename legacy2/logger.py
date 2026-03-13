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
        
        # Initialize CSV with headers
        self.headers = [
            "epoch", "behavior", "adversary_attack", 
            "target_response", "reward", "is_jailbroken", "explanation"
        ]
        with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()

    def log_step(self, data: Dict[str, Any]):
        with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(data)