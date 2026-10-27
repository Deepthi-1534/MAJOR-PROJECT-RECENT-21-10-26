from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MODEL_DIR = Path("models")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 64
STRIDE = 16
BATCH_SIZE = 128

THRESHOLD = 0.60

MIN_AREA = 8


# ==================================================
# CNN BRANCH
# ==================================================

class CNNBranch(nn.Module):

    def __init__(self, input_channels):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                input_channels,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):

        x = self.features(x)

        return x.flatten(1)


# ==================================================
# SINGLE-MODALITY CNN
# ==================================================

class FeatureCNN(nn.Module):

    def __init__(self, input_channels):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                input_channels,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(
            128,
            1
        )

    def forward(self, x):

        x = self.features(x)

        x = x.flatten(1)

        return self.classifier(x).squeeze(1)


# ==================================================
# MULTIMODAL CNN
# ==================================================

class MultimodalCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.rgb_branch = CNNBranch(3)

        self.lwir_branch = CNNBranch(1)

        self.classifier = nn.Sequential(

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                1
            )
        )

    def forward(self, rgb, lwir):

        rgb_features = self.rgb_branch(rgb)

        lwir_features = self.lwir_branch(lwir)

        combined = torch.cat(
            [
                rgb_features,
                lwir_features
            ],
            dim=1
        )

        return self.classifier(
            combined
        ).squeeze(1)


# ==================================================
# FEATURE AGENT
# ==================================================

class FeatureAgent:

    def __init__(self):

        self.rgb_model = None
        self.lwir_model = None
        self.multimodal_model = None

        print(
            f"Feature Agent device: {DEVICE}"
        )

        # ------------------------------------------
        # RGB CNN
        # ------------------------------------------

        rgb_path = MODEL_DIR / "rgb_cnn.pt"

        if rgb_path.exists():

            self.rgb_model = FeatureCNN(3)

            self.rgb_model.load_state_dict(
                torch.load(
                    rgb_path,
                    map_location=DEVICE
                )
            )

            self.rgb_model.to(DEVICE)
            self.rgb_model.eval()

        # ------------------------------------------
        # LWIR CNN
        # ------------------------------------------

        lwir_path = MODEL_DIR / "lwir_cnn.pt"

        if lwir_path.exists():

            self.lwir_model = FeatureCNN(1)

            self.lwir_model.load_state_dict(
                torch.load(
                    lwir_path,
                    map_location=DEVICE
                )
            )

            self.lwir_model.to(DEVICE)
            self.lwir_model.eval()

        # ------------------------------------------
        # MULTIMODAL CNN
        # ------------------------------------------

        multimodal_path = (
            MODEL_DIR /
            "multimodal_cnn.pt"
        )

        if multimodal_path.exists():

            self.multimodal_model = MultimodalCNN()

            self.multimodal_model.load_state_dict(
                torch.load(
                    multimodal_path,
                    map_location=DEVICE
                )
            )

            self.multimodal_model.to(DEVICE)
            self.multimodal_model.eval()

    # ==================================================
    # RGB PATCH PREPARATION
    # ==================================================

    def prepare_rgb_patch(self, patch):

        patch = cv2.resize(
            patch,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        patch = cv2.cvtColor(
            patch,
            cv2.COLOR_BGR2RGB
        )

        patch = (
            patch.astype(np.float32) /
            255.0
        )

        return torch.from_numpy(
            patch
        ).permute(2, 0, 1)

    # ==================================================
    # LWIR PATCH PREPARATION
    # ==================================================

    def prepare_lwir_patch(self, patch):

        patch = cv2.resize(
            patch,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        patch = (
            patch.astype(np.float32) /
            255.0
        )

        return torch.from_numpy(
            patch
        ).unsqueeze(0)

    # ==================================================
    # GENERATE PATCHES
    # ==================================================

    def generate_patches(self, image):

        height, width = image.shape[:2]

        patches = []
        boxes = []

        y_positions = range(
            0,
            max(
                height - IMAGE_SIZE + 1,
                1
            ),
            STRIDE
        )

        x_positions = range(
            0,
            max(
                width - IMAGE_SIZE + 1,
                1
            ),
            STRIDE
        )

        for y in y_positions:

            for x in x_positions:

                x2 = min(
                    x + IMAGE_SIZE,
                    width
                )

                y2 = min(
                    y + IMAGE_SIZE,
                    height
                )

                patch = image[
                    y:y2,
                    x:x2
                ]

                if (
                    patch.shape[0] != IMAGE_SIZE
                    or
                    patch.shape[1] != IMAGE_SIZE
                ):
                    continue

                patches.append(patch)

                boxes.append(
                    [
                        x,
                        y,
                        IMAGE_SIZE,
                        IMAGE_SIZE
                    ]
                )

        return patches, boxes

    # ==================================================
    # BATCH PREDICTION
    # ==================================================

    def predict_batches(
        self,
        rgb_patches=None,
        lwir_patches=None
    ):

        probabilities = []

        if rgb_patches is not None:

            total = len(rgb_patches)

        else:

            total = len(lwir_patches)

        for start in range(
            0,
            total,
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                total
            )

            # --------------------------------------
            # RGB + LWIR
            # --------------------------------------

            if (
                rgb_patches is not None
                and
                lwir_patches is not None
            ):

                if self.multimodal_model is None:

                    raise RuntimeError(
                        "Multimodal CNN model is not available."
                    )

                rgb_batch = torch.stack(
                    [
                        self.prepare_rgb_patch(p)
                        for p in rgb_patches[start:end]
                    ]
                ).to(DEVICE)

                lwir_batch = torch.stack(
                    [
                        self.prepare_lwir_patch(p)
                        for p in lwir_patches[start:end]
                    ]
                ).to(DEVICE)

                with torch.no_grad():

                    output = self.multimodal_model(
                        rgb_batch,
                        lwir_batch
                    )

            # --------------------------------------
            # RGB ONLY
            # --------------------------------------

            elif rgb_patches is not None:

                if self.rgb_model is None:

                    raise RuntimeError(
                        "RGB CNN model is not available."
                    )

                rgb_batch = torch.stack(
                    [
                        self.prepare_rgb_patch(p)
                        for p in rgb_patches[start:end]
                    ]
                ).to(DEVICE)

                with torch.no_grad():

                    output = self.rgb_model(
                        rgb_batch
                    )

            # --------------------------------------
            # LWIR ONLY
            # --------------------------------------

            else:

                if self.lwir_model is None:

                    raise RuntimeError(
                        "LWIR CNN model is not available."
                    )

                lwir_batch = torch.stack(
                    [
                        self.prepare_lwir_patch(p)
                        for p in lwir_patches[start:end]
                    ]
                ).to(DEVICE)

                with torch.no_grad():

                    output = self.lwir_model(
                        lwir_batch
                    )

            batch_probability = (
                torch.sigmoid(output)
                .cpu()
                .numpy()
            )

            probabilities.extend(
                batch_probability.tolist()
            )

        return np.array(
            probabilities
        )

    # ==================================================
    # ANALYZE
    # ==================================================

    def analyze(
        self,
        rgb_path=None,
        lwir_path=None
    ):

        if (
            rgb_path is None
            and
            lwir_path is None
        ):

            raise ValueError(
                "RGB or LWIR image is required."
            )

        rgb_image = None
        lwir_image = None

        # ------------------------------------------
        # LOAD RGB
        # ------------------------------------------

        if rgb_path is not None:

            rgb_image = cv2.imread(
                str(rgb_path)
            )

            if rgb_image is None:

                raise ValueError(
                    f"Could not load RGB image: {rgb_path}"
                )

        # ------------------------------------------
        # LOAD LWIR
        # ------------------------------------------

        if lwir_path is not None:

            lwir_image = cv2.imread(
                str(lwir_path),
                cv2.IMREAD_GRAYSCALE
            )

            if lwir_image is None:

                raise ValueError(
                    f"Could not load LWIR image: {lwir_path}"
                )

        # ------------------------------------------
        # DETERMINE MODALITY
        # ------------------------------------------

        if (
            rgb_image is not None
            and
            lwir_image is not None
        ):

            modality = "rgb+lwir"

            height, width = rgb_image.shape[:2]

            lwir_image = cv2.resize(
                lwir_image,
                (width, height)
            )

            rgb_patches, boxes = (
                self.generate_patches(
                    rgb_image
                )
            )

            lwir_patches, _ = (
                self.generate_patches(
                    lwir_image
                )
            )

            probabilities = (
                self.predict_batches(
                    rgb_patches,
                    lwir_patches
                )
            )

        elif rgb_image is not None:

            modality = "rgb"

            rgb_patches, boxes = (
                self.generate_patches(
                    rgb_image
                )
            )

            probabilities = (
                self.predict_batches(
                    rgb_patches=rgb_patches
                )
            )

        else:

            modality = "lwir"

            lwir_patches, boxes = (
                self.generate_patches(
                    lwir_image
                )
            )

            probabilities = (
                self.predict_batches(
                    lwir_patches=lwir_patches
                )
            )

        # ------------------------------------------
        # CREATE PROBABILITY MAP
        # ------------------------------------------

        if modality in [
            "rgb",
            "rgb+lwir"
        ]:

            height, width = (
                rgb_image.shape[:2]
            )

        else:

            height, width = (
                lwir_image.shape[:2]
            )

        probability_map = np.zeros(
            (height, width),
            dtype=np.float32
        )

        count_map = np.zeros(
            (height, width),
            dtype=np.float32
        )

        for probability, box in zip(
            probabilities,
            boxes
        ):

            x, y, w, h = box

            probability_map[
                y:y+h,
                x:x+w
            ] += probability

            count_map[
                y:y+h,
                x:x+w
            ] += 1

        valid = count_map > 0

        probability_map[valid] /= (
            count_map[valid]
        )

        # ------------------------------------------
        # DETECTION MASK
        # ------------------------------------------

        detection_mask = (
            probability_map >= THRESHOLD
        ).astype(np.uint8) * 255

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        detection_mask = cv2.morphologyEx(
            detection_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        detection_mask = cv2.morphologyEx(
            detection_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # ------------------------------------------
        # CONNECTED COMPONENTS
        # ------------------------------------------

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                detection_mask,
                connectivity=8
            )
        )

        regions = []

        for i in range(
            1,
            num_labels
        ):

            x = stats[
                i,
                cv2.CC_STAT_LEFT
            ]

            y = stats[
                i,
                cv2.CC_STAT_TOP
            ]

            w = stats[
                i,
                cv2.CC_STAT_WIDTH
            ]

            h = stats[
                i,
                cv2.CC_STAT_HEIGHT
            ]

            area = stats[
                i,
                cv2.CC_STAT_AREA
            ]

            if area < MIN_AREA:
                continue

            region_mask = (
                labels == i
            )

            region_probability = (
                probability_map[
                    region_mask
                ]
            )

            regions.append({

                "bbox": [
                    int(x),
                    int(y),
                    int(w),
                    int(h)
                ],

                "area":
                    int(area),

                "evidence":
                    float(
                        region_probability.mean()
                    ),

                "max_probability":
                    float(
                        region_probability.max()
                    )
            })

        regions.sort(
            key=lambda r: r["evidence"],
            reverse=True
        )

       

        # ------------------------------------------
        # ------------------------------------------
        # OVERALL EVIDENCE
        # ------------------------------------------

        if len(probabilities) > 0:

            overall_evidence = float(
                np.mean(probabilities)
            )

            max_probability = float(
                np.max(probabilities)
            )

            max_index = int(
                np.argmax(probabilities)
            )

        else:

            overall_evidence = 0.0
            max_probability = 0.0
            max_index = -1

        # ------------------------------------------
        # LOCALIZED FEATURE EVIDENCE
        # ------------------------------------------

        localized_evidence_regions = []

        # Preserve a strong local CNN response even
        # when it does not survive final candidate
        # region post-processing.

        LOCAL_EVIDENCE_THRESHOLD = 0.75

        if (
            max_index >= 0
            and
            max_probability >= LOCAL_EVIDENCE_THRESHOLD
        ):

            peak_box = boxes[max_index]

            localized_evidence_regions.append({

                "bbox": [
                    int(peak_box[0]),
                    int(peak_box[1]),
                    int(peak_box[2]),
                    int(peak_box[3])
                ],

                "area": int(
                    peak_box[2] *
                    peak_box[3]
                ),

                "evidence":
                    max_probability,

                "max_probability":
                    max_probability,

                "evidence_type":
                    "localized"
            })

        # ------------------------------------------
        # AGENT STATUS
        # ------------------------------------------

        if len(regions) > 0:

            status = "DETECTED"

        elif len(localized_evidence_regions) > 0:

            status = "EVIDENCE_PRESENT"

        else:

            status = "NO_DETECTION"

        # ------------------------------------------
        # RESULT
        # ------------------------------------------

        return {

            "agent":
                "feature",

            "status":
                status,

            "modality":
                modality,

            "rgb_available":
                rgb_image is not None,

            "lwir_available":
                lwir_image is not None,

            "overall_evidence":
                overall_evidence,

            "max_probability":
                max_probability,

            "regions":
                regions,

            "localized_evidence_regions":
                localized_evidence_regions,

            "probability_map":
                probability_map,

            "num_patches":
                len(probabilities)
        }

        # ------------------------------------------
        # AGENT STATUS
        # ------------------------------------------

        if len(regions) > 0:

            status = "DETECTED"

        else:

            status = "NO_DETECTION"

        # ------------------------------------------
        # RESULT
        # ------------------------------------------

        return {

            "agent":
                "feature",

            "status":
                status,

            "modality":
                modality,

            "rgb_available":
                rgb_image is not None,

            "lwir_available":
                lwir_image is not None,

            "overall_evidence":
                overall_evidence,

            "regions":
                regions,

            "probability_map":
                probability_map,

            "num_patches":
                len(probabilities)
        }

    # ==================================================
    # SAVE VISUALIZATION
    # ==================================================

    def save_visualization(
        self,
        rgb_path=None,
        lwir_path=None,
        output_path="outputs/feature_analysis.png"
    ):

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        result = self.analyze(
            rgb_path=rgb_path,
            lwir_path=lwir_path
        )

        probability_map = result[
            "probability_map"
        ]

        modality = result[
            "modality"
        ]

        status = result[
            "status"
        ]

        # ------------------------------------------
        # DISPLAY IMAGE
        # ------------------------------------------

        if rgb_path is not None:

            display_image = cv2.imread(
                str(rgb_path)
            )

        else:

            display_image = cv2.imread(
                str(lwir_path),
                cv2.IMREAD_GRAYSCALE
            )

            display_image = cv2.cvtColor(
                display_image,
                cv2.COLOR_GRAY2BGR
            )

        if display_image is None:

            raise ValueError(
                "Could not load image for visualization."
            )

        # ------------------------------------------
        # HEATMAP
        # ------------------------------------------

        heatmap = (
            np.clip(
                probability_map,
                0,
                1
            ) * 255
        ).astype(np.uint8)

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        # ------------------------------------------
        # OVERLAY
        # ------------------------------------------

        overlay = cv2.addWeighted(
            display_image,
            0.55,
            heatmap,
            0.45,
            0
        )

        # ------------------------------------------
        # REGIONS
        # ------------------------------------------

        regions_image = display_image.copy()

        for index, region in enumerate(
            result["regions"][:10],
            start=1
        ):

            x, y, w, h = region["bbox"]

            cv2.rectangle(
                regions_image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            label = (
                f"R{index} "
                f"E={region['evidence']:.2f}"
            )

            cv2.putText(
                regions_image,
                label,
                (
                    x,
                    max(y - 8, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        # ------------------------------------------
        # TITLES
        # ------------------------------------------

        def add_title(
            image,
            title
        ):

            canvas = image.copy()

            cv2.rectangle(
                canvas,
                (0, 0),
                (
                    canvas.shape[1],
                    38
                ),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                canvas,
                title,
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            return canvas

        display_image = add_title(
            display_image,
            f"INPUT ({modality.upper()})"
        )

        heatmap = add_title(
            heatmap,
            "CNN FEATURE EVIDENCE"
        )

        overlay = add_title(
            overlay,
            "FEATURE EVIDENCE OVERLAY"
        )

        regions_image = add_title(
            regions_image,
            "FEATURE CANDIDATE REGIONS"
        )

        # ------------------------------------------
        # SUMMARY
        # ------------------------------------------

        summary_height = 145

        summary = np.zeros(
            (
                summary_height,
                display_image.shape[1] * 2,
                3
            ),
            dtype=np.uint8
        )

        summary_lines = [

    f"FEATURE AGENT",

    f"Mode: {modality.upper()}",

    f"Status: "
    f"{result['status']}",

    f"Patches scanned: "
    f"{result['num_patches']}",

    f"Overall evidence: "
    f"{result['overall_evidence']:.3f}",

    f"Maximum probability: "
    f"{result['max_probability']:.3f}",

    f"Candidate regions: "
    f"{len(result['regions'])}"
]

        y_position = 22

        for line in summary_lines:

            cv2.putText(
                summary,
                line,
                (15, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            y_position += 18

        # ------------------------------------------
        # GRID
        # ------------------------------------------

        top_row = np.hstack([
            display_image,
            heatmap
        ])

        bottom_row = np.hstack([
            overlay,
            regions_image
        ])

        grid = np.vstack([
            top_row,
            bottom_row
        ])

        final_visualization = np.vstack([
            grid,
            summary
        ])

        cv2.imwrite(
            str(output_path),
            final_visualization
        )

        return output_path


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    dataset = Path(
        "MUDCAD-X/dataset"
    )

    rgb_path = next(
        dataset.rglob("vis.png")
    )

    lwir_path = (
        rgb_path.parent /
        "lwir.png"
    )

    agent = FeatureAgent()

    print("\nFEATURE AGENT")
    print("=========================")

    result = agent.analyze(
        rgb_path=rgb_path,
        lwir_path=lwir_path
    )

    print(
        f"Modality: "
        f"{result['modality']}"
    )

    print(
        f"Status: "
        f"{result['status']}"
    )

    print(
        f"RGB available: "
        f"{result['rgb_available']}"
    )

    print(
        f"LWIR available: "
        f"{result['lwir_available']}"
    )

    print(
        f"Patches scanned: "
        f"{result['num_patches']}"
    )

    print(
        f"Overall feature evidence: "
        f"{result['overall_evidence']:.3f}"
    )

    print(
        f"Maximum feature probability: "
        f"{result.get('max_probability', 0.0):.3f}"
    )

    print(
        f"Feature regions: "
        f"{len(result['regions'])}"
    )

    print(
        f"Localized evidence regions: "
        f"{len(result.get('localized_evidence_regions', []))}"
    )

    for i, region in enumerate(
        result.get("localized_evidence_regions", [])[:10],
        start=1
    ):

        print(
            f"\\nLocalized Evidence {i}"
        )

        print(
            f"  Box: "
            f"{region['bbox']}"
        )

        print(
            f"  Evidence: "
            f"{region['evidence']:.3f}"
        )

        print(
            f"  Max probability: "
            f"{region['max_probability']:.3f}"
        )

    for i, region in enumerate(
        result["regions"][:10],
        start=1
    ):

        print(
            f"\nRegion {i}"
        )

        print(
            f"  Box: "
            f"{region['bbox']}"
        )

        print(
            f"  Area: "
            f"{region['area']}"
        )

        print(
            f"  Evidence: "
            f"{region['evidence']:.3f}"
        )

        print(
            f"  Max probability: "
            f"{region['max_probability']:.3f}"
        )

    visualization_path = (
        agent.save_visualization(
            rgb_path=rgb_path,
            lwir_path=lwir_path
        )
    )

    print(
        f"\nVisualization saved to:"
        f"\n{visualization_path}"
    )