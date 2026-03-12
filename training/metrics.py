import json
from pathlib import Path

def save_history(history, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    serializable = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)