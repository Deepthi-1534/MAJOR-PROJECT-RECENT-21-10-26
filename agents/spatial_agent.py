from pathlib import Path
import argparse

import cv2
from ultralytics import YOLO


class SpatialAgent:
    """Run YOLO inference on RGB/VIS images for spatial evidence.

    The defaults use a lower confidence threshold and a larger inference
    size to make a diagnostic pass for small or low-contrast candidates.
    These settings can increase false positives and should be validated.
    """

    def __init__(
        self,
        model_path="runs/detect/spatial_train-3/weights/best.pt",
        confidence_threshold=0.15,
        image_size=768,
        device="cpu",
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.image_size = int(image_size)
        self.device = str(device)

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model weights not found: {self.model_path}\n"
                "Check that the model file exists at the specified path."
            )

        print("Loading Spatial Agent...")
        print(f"Model: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Confidence threshold: {self.confidence_threshold:.2f}")
        print(f"Inference image size: {self.image_size}")

        self.model = YOLO(str(self.model_path))
        print("Spatial Agent loaded successfully.")

    def analyze(self, image_path):
        """Run YOLO and return detections plus summary statistics."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"RGB image not found: {image_path}")

        prediction = self.model.predict(
            source=str(image_path),
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        if prediction.boxes is not None:
            for box in prediction.boxes:
                x1, y1, x2, y2 = (
                    box.xyxy[0].detach().cpu().numpy().tolist()
                )
                confidence = float(box.conf[0].detach().cpu().item())
                class_id = int(box.cls[0].detach().cpu().item())
                class_names = self.model.names
                class_name = (
                    class_names.get(class_id, str(class_id))
                    if isinstance(class_names, dict)
                    else class_names[class_id]
                )

                width = max(0.0, float(x2 - x1))
                height = max(0.0, float(y2 - y1))
                detections.append(
                    {
                        "class": class_name,
                        "class_id": class_id,
                        # xywh format: top-left x/y, width, height
                        "bbox": [float(x1), float(y1), width, height],
                        "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": confidence,
                        "area": width * height,
                    }
                )

        count = len(detections)
        average_confidence = (
            sum(d["confidence"] for d in detections) / count if count else 0.0
        )
        maximum_confidence = (
            max(d["confidence"] for d in detections) if count else 0.0
        )
        class_counts = {}
        for detection in detections:
            name = detection["class"]
            class_counts[name] = class_counts.get(name, 0) + 1

        return {
            "agent": "spatial",
            "image": str(image_path),
            "detections": detections,
            "detection_count": count,
            "average_confidence": average_confidence,
            "maximum_confidence": maximum_confidence,
            "class_counts": class_counts,
            "confidence_threshold": self.confidence_threshold,
            "image_size": self.image_size,
        }

    def save_visualization(
        self,
        image_path,
        output_path="outputs/spatial_detection.png",
    ):
        """Save a YOLO-annotated image and return its path."""
        image_path = Path(image_path)
        output_path = Path(output_path)
        if not image_path.exists():
            raise FileNotFoundError(f"RGB image not found: {image_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        results = self.model.predict(
            source=str(image_path),
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        if not results:
            raise RuntimeError(f"YOLO returned no results for: {image_path}")

        annotated_image = results[0].plot()
        success = cv2.imwrite(str(output_path), annotated_image)
        if not success:
            raise OSError(f"Failed to save visualization: {output_path}")

        print(f"Spatial visualization saved to: {output_path}")
        return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the Spatial Agent on an RGB/VIS image."
    )
    parser.add_argument("--image", required=True, help="Path to the RGB/VIS image.")
    parser.add_argument(
        "--model",
        default="runs/detect/spatial_train-3/weights/best.pt",
        help="Path to trained YOLO weights.",
    )
    parser.add_argument(
        "--output",
        default="outputs/spatial_detection.png",
        help="Path for the annotated output image.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "0"],
        help="Inference device: cpu or CUDA GPU 0.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Detection confidence threshold (default: 0.15).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=768,
        help="YOLO inference image size (default: 768).",
    )
    args = parser.parse_args()

    agent = SpatialAgent(
        model_path=args.model,
        confidence_threshold=args.conf,
        image_size=args.imgsz,
        device=args.device,
    )
    result = agent.analyze(args.image)

    print("\nSPATIAL AGENT RESULTS")
    print("=" * 50)
    print(f"Image: {result['image']}")
    print(f"Detections: {result['detection_count']}")
    print(f"Average confidence: {result['average_confidence']:.4f}")
    print(f"Maximum confidence: {result['maximum_confidence']:.4f}")
    print(f"Class counts: {result['class_counts']}")
    print(f"Confidence threshold: {result['confidence_threshold']:.2f}")
    print(f"Inference image size: {result['image_size']}")

    for i, detection in enumerate(result["detections"], start=1):
        print(
            f"{i:02d}. {detection['class']} "
            f"confidence={detection['confidence']:.3f} "
            f"bbox_xywh={[round(v, 1) for v in detection['bbox']]}"
        )

    output_path = agent.save_visualization(args.image, args.output)
    print(f"Visualization: {output_path}")
