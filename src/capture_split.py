from pathlib import Path
import json
import random



# SETTINGS


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_DIR
    / "MUDCAD-X"
    / "dataset"
)

SPLIT_FILE = (
    PROJECT_DIR
    / "data"
    / "capture_splits.json"
)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42 # controls the pseudorandom shuffle: makes the shuffle reproducible when its like in the same order capture list


# ===================
# FIND COMPLETE CAPTURES


def find_complete_captures():

    captures = []

    for label_path in DATASET_DIR.rglob("label.png"):

        capture_dir = label_path.parent

        rgb_path = capture_dir / "vis.png"
        lwir_path = capture_dir / "lwir.png"

        if (
            rgb_path.exists()
            and lwir_path.exists()
        ):

            captures.append(
                capture_dir
            )

  

    captures = sorted(
        captures,
        key=lambda p: str(p).lower()
    )

    return captures


# CREATE  MASTER CAPTURE SPLIT


def create_split():

    captures = find_complete_captures()

    if not captures:
        raise FileNotFoundError(
            "No complete RGB + LWIR captures found."
        )

    print(
        f"Complete captures found: "
        f"{len(captures)}"
    )

    shuffled = captures.copy()

    random.Random(SEED).shuffle(
        shuffled
    )

    total = len(shuffled)

    train_end = int(
        total * TRAIN_RATIO
    )

    val_end = (
        train_end
        + int(total * VAL_RATIO)
    )

    train = shuffled[
        :train_end
    ]

    val = shuffled[
        train_end:val_end
    ]

    test = shuffled[
        val_end:
    ]

    split_data = {

        "seed": SEED,

        "ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO
        },

        "counts": {
            "total": total,
            "train": len(train),
            "val": len(val),
            "test": len(test)
        },

        "train": [
            str(
                path.relative_to(
                    PROJECT_DIR
                ).as_posix()
            )
            for path in train
        ],

        "val": [
            str(
                path.relative_to(
                    PROJECT_DIR
                ).as_posix()
            )
            for path in val
        ],

        "test": [
            str(
                path.relative_to(
                    PROJECT_DIR
                ).as_posix()
            )
            for path in test
        ]
    }

    SPLIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        SPLIT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            split_data,
            file,
            indent=2
        )

    print()
    print("=" * 60)
    print("CAPTURE SPLIT CREATED")
    print("=" * 60)

    print(
        f"Train: {len(train)}"
    )

    print(
        f"Validation: {len(val)}"
    )

    print(
        f"Test: {len(test)}"
    )

    print(
        f"Total: {total}"
    )

    print()
    print(
        f"Saved to: {SPLIT_FILE}"
    )

    return split_data


# ==================
# LOAD EXISTING SPLIT


def load_split():

    if not SPLIT_FILE.exists():

        print(
            "Capture split file not found."
        )

        print(
            "Creating a new split..."
        )

        return create_split()

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        split_data = json.load(
            file
        )

    return split_data


# ============================================================
# GET CAPTURES FOR A SPLIT


def get_split_captures(
    split_name
):

    if split_name not in {
        "train",
        "val",
        "test"
    }:

        raise ValueError(
            "split_name must be "
            "'train', 'val', or 'test'."
        )

    split_data = load_split()

    captures = []

    for relative_path in split_data[
        split_name
    ]:

        capture_path = (
            PROJECT_DIR
            / relative_path
        )

        if not capture_path.exists():

            raise FileNotFoundError(
                f"Capture does not exist: "
                f"{capture_path}"
            )

        captures.append(
            capture_path
        )

    return captures


# ====
# GET SPLIT FOR A CAPTURE


def get_capture_split(
    capture_path
):

    capture_path = Path(
        capture_path
    ).resolve()

    split_data = load_split()

    relative_path = str(
        capture_path.relative_to(
            PROJECT_DIR
        ).as_posix()
    )

    for split_name in [
        "train",
        "val",
        "test"
    ]:

        if relative_path in split_data[
            split_name
        ]:

            return split_name

    return None


# ============================================================
# MAIN
# 

if __name__ == "__main__":

    create_split()