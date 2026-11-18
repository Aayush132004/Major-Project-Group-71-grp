"""Start the PlannerLLM console:  python scripts/serve.py  ->  http://127.0.0.1:8000"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"PlannerLLM console on http://127.0.0.1:{port}")
    uvicorn.run("server.app:app", host="127.0.0.1", port=port, log_level="warning")
