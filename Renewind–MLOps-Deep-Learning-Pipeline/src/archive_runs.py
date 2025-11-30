# src/archive.py

import os
import shutil
from datetime import datetime
import yaml

def get_paths():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_cfg = config["artifacts"]

    artifacts_dir = os.path.join(base_dir, artifacts_cfg["base_dir"])
    current_dir   = os.path.join(artifacts_dir, artifacts_cfg["current_subdir"])
    archive_dir   = os.path.join(artifacts_dir, artifacts_cfg["archive_subdir"])

    return artifacts_dir, current_dir, archive_dir

def archive_current_run(run_id: str | None = None) -> str:
    artifacts_dir, current_dir, archive_dir = get_paths()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")
    ingested_dir  = os.path.join(base_dir, "data", "ingested")

    # generate a timestamp-based run_id if not provided
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_archive_dir = os.path.join(archive_dir, run_id)

    # Nothing to do if current is empty / missing
    if not os.path.exists(current_dir) or not os.listdir(current_dir):
        print(f"[archive] Nothing to archive, '{current_dir}' is empty.")
        return run_id

    os.makedirs(archive_dir, exist_ok=True)

    # Move current -> archive/<run_id>
    print(f"[archive] Archiving '{current_dir}' -> '{run_archive_dir}'")
    shutil.move(current_dir, run_archive_dir)

    # Recreate empty 'current'
    os.makedirs(current_dir, exist_ok=True)
    print(f"[archive] Recreated empty working dir '{current_dir}'")

    for d in [processed_dir, ingested_dir]:
        if os.path.exists(d):
            print(f"[archive] Clearing directory '{d}'")
            shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

    # Move raw CSV into archive and restore clean copy
    raw_file = os.path.join(base_dir, "data", "raw", "Renewind.csv")
    raw_dest = os.path.join(run_archive_dir, "Renewind.csv")
    if os.path.exists(raw_file):
        print(f"[archive] Moving raw data '{raw_file}' -> '{raw_dest}'")
        shutil.move(raw_file, raw_dest)
        # Optionally recreate empty raw folder or restore template file
        os.makedirs(os.path.join(base_dir, "data", "raw"), exist_ok=True)

    return run_id

if __name__ == "__main__":
    archive_current_run()