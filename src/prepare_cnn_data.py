from pathlib import Path
import json

import cv2
import numpy as np


# --------------------------------------------------
# PATHS


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "MUDCAD-X" / "dataset"
OUTPUT_DIR = PROJECT_DIR / "data" / "cnn"

SPLIT_FILE = PROJECT_DIR / "data" / "capture_splits.json"

PATCH_SIZE = 64
PATCHES_PER_CLASS = 20

SEED = 42


# --------------------------------------------------
# LOAD MASTER CAPTURE SPLIT
# -----------------------------

if not SPLIT_FILE.exists():

    raise FileNotFoundError(
        f"Master split file not found:\n"
        f"{SPLIT_FILE}\n\n"
        f"Run src/capture_split.py first."
    )


with open(
    SPLIT_FILE,
    "r",
    encoding="utf-8"
) as f:

    split_data = json.load(f)


# ---------------------------------------------------------------
# LOAD CAPTURES FROM    SPLIT


def load_split_captures(split_name):

    captures = []

    for relative_path in split_data[split_name]:

        capture = (
            PROJECT_DIR / relative_path
        )

        vis_path = capture / "vis.png"
        lwir_path = capture / "lwir.png"
        label_path = capture / "label.png"

        if not (
            vis_path.exists()
            and lwir_path.exists()
            and label_path.exists()
        ):

            raise FileNotFoundError(
                f"Incomplete capture:\n"
                f"{capture}"
            )

        captures.append(capture)

    return captures


splits = {

    "train": load_split_captures(
        "train"
    ),

    "val": load_split_captures(
        "val"
        
    ),

    "test": load_split_captures(
        "test"
    )
}


# --------------------------------------------------
# PRINT SPLIT INFORMATION


print("=" * 60)
print("USING MASTER CAPTURE SPLIT")
print("=" * 60)

print(
    f"Split file: {SPLIT_FILE}"
)

print(
    f"Seed: {split_data['seed']}"
)

print()

for split_name, captures in splits.items():

    print(
        f"{split_name}: "
        f"{len(captures)} captures"
    )

print()


# --------------------------------------------------
# CREATE PATCH


def extract_patch(
    image,
    x,
    y,
    size
):

    half = size // 2

    h, w = image.shape[:2]

    x1 = max(
        0,
        x - half
    )

    y1 = max(
        0,
        y - half
    )

    x2 = min(
        w,
        x + half
    )

    y2 = min(
        h,
        y + half
    )

    patch = image[
        y1:y2,
        x1:x2
    ]

    if (
        patch.shape[0] != size
        or
        patch.shape[1] != size
    ):

        patch = cv2.copyMakeBorder(
            patch,
            0,
            size - patch.shape[0],
            0,
            size - patch.shape[1],
            cv2.BORDER_REFLECT
        )

    return patch


# --------------------------------------------------
# PROCESS CAPTURE
# --------------------------------------------------

def process_capture(
    capture,
    output_dir,
    capture_id
):

    vis = cv2.imread(
        str(capture / "vis.png")
    )

    lwir = cv2.imread(
        str(capture / "lwir.png"),
        cv2.IMREAD_GRAYSCALE
    )

    label = cv2.imread(
        str(capture / "label.png")
    )

    if (
        vis is None
        or
        lwir is None
        or
        label is None
    ):

        return 0


    # --------------------------------------------------
    # CREATE TARGET / BACKGROUND MASKS
   

    # Any non-black label pixel is
    # considered a target.

    target_mask = np.any(
        label != 0,
        axis=2
    )

    background_mask = ~target_mask


    target_y, target_x = np.where(
        target_mask
    )

    background_y, background_x = np.where(
        background_mask
    )


    if len(target_x) == 0:

        return 0


    # --------------------------------------------------
    # RANDOM PATCH SAMPLING
    # --------------------------------------------------

    rng = np.random.default_rng(
        SEED + capture_id
    )


    target_count = min(
        PATCHES_PER_CLASS,
        len(target_x)
    )

    background_count = min(
        PATCHES_PER_CLASS,
        len(background_x)
    )


    target_indices = rng.choice(
        len(target_x),
        target_count,
        replace=False
    )

    background_indices = rng.choice(
        len(background_x),
        background_count,
        replace=False
    )


    count = 0


    # --------------------------------------------------
    # TARGET PATCHES
    # --------------------------------------------------

    for i in target_indices:

        x = int(
            target_x[i]
        )

        y = int(
            target_y[i]
        )


        vis_patch = extract_patch(
            vis,
            x,
            y,
            PATCH_SIZE
        )


        lwir_patch = extract_patch(
            lwir,
            x,
            y,
            PATCH_SIZE
        )


        filename = (
            f"{capture_id:04d}_"
            f"target_{count:03d}"
        )


        cv2.imwrite(
            str(
                output_dir
                / "target"
                / f"{filename}_rgb.jpg"
            ),
            vis_patch
        )


        cv2.imwrite(
            str(
                output_dir
                / "target"
                / f"{filename}_lwir.jpg"
            ),
            lwir_patch
        )


        count += 1


    # --------------------------------------------------
    # BACKGROUND PATCHES
    # --------------------------------------------------

    for i in background_indices:

        x = int(
            background_x[i]
        )

        y = int(
            background_y[i]
        )


        vis_patch = extract_patch(
            vis,
            x,
            y,
            PATCH_SIZE
        )


        lwir_patch = extract_patch(
            lwir,
            x,
            y,
            PATCH_SIZE
        )


        filename = (
            f"{capture_id:04d}_"
            f"background_{i:03d}"
        )


        cv2.imwrite(
            str(
                output_dir
                / "background"
                / f"{filename}_rgb.jpg"
            ),
            vis_patch
        )


        cv2.imwrite(
            str(
                output_dir
                / "background"
                / f"{filename}_lwir.jpg"
            ),
            lwir_patch
        )


        count += 1


    return count


# --------------------------------------------------
# PROCESS SPLIT
# --------------------------------------------------

def process_split(
    captures,
    split_name
):

    output_dir = (
        OUTPUT_DIR
        / split_name
    )


    target_dir = (
        output_dir
        / "target"
    )


    background_dir = (
        output_dir
        / "background"
    )


    target_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    background_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    total = 0


    for i, capture in enumerate(
        captures
    ):

        count = process_capture(
            capture,
            output_dir,
            i
        )


        total += count


        if (
            (i + 1) % 100 == 0
        ):

            print(
                f"  {split_name}: "
                f"{i + 1}/"
                f"{len(captures)}"
            )


    print(
        f"  {split_name} patches: "
        f"{total}"
    )


    return total


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print(
        "Preparing multimodal CNN dataset..."
    )


    print(
        "\nCapture split:"
    )

    for split_name, captures in (
        splits.items()
    ):

        print(
            f"{split_name}: "
            f"{len(captures)}"
        )


    print()


    # --------------------------------------------------
    # PROCESS TRAIN
    # --------------------------------------------------

    train_patches = process_split(
        splits["train"],
        "train"
    )


    # --------------------------------------------------
    # PROCESS VALIDATION
    # --------------------------------------------------

    val_patches = process_split(
        splits["val"],
        "val"
    )


    # --------------------------------------------------
    # PROCESS TEST
    # --------------------------------------------------

    test_patches = process_split(
        splits["test"],
        "test"
    )


    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("CNN DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print(
        f"Train captures: "
        f"{len(splits['train'])}"
    )

    print(
        f"Validation captures: "
        f"{len(splits['val'])}"
    )

    print(
        f"Test captures: "
        f"{len(splits['test'])}"
    )

    print()

    print(
        f"Train patches: "
        f"{train_patches}"
    )

    print(
        f"Validation patches: "
        f"{val_patches}"
    )

    print(
        f"Test patches: "
        f"{test_patches}"
    )

    print()

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()