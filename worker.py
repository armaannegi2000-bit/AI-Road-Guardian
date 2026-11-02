"""Run road-image analysis jobs in a process separate from Flask."""

import os

# Prevent importing app.py from also starting its in-process worker thread.
os.environ["AI_ANALYSIS_WORKER_MODE"] = "external"

from app import _ai_analysis_worker


def main():
    print("Road Guardian AI worker started. Press Ctrl+C to stop.")
    try:
        _ai_analysis_worker()
    except KeyboardInterrupt:
        print("Road Guardian AI worker stopped.")


if __name__ == "__main__":
    main()
