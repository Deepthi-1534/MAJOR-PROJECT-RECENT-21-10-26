
import subprocess
import sys
from pathlib import Path

TOTAL_IMAGES = 10  # Start by testing 10 images

for index in range(1, TOTAL_IMAGES + 1):
    print(f"\nProcessing image {index}/{TOTAL_IMAGES}")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agents.rev1_pipeline",
            "--index",
            str(index),
        ],
        check=False,
    )

    if result.returncode != 0:
        print(f"Image {index} failed. Check the error above.")
    else:
        print(f"Image {index} completed.")

print("\nBatch processing finished.")
