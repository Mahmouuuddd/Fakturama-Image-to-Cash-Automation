from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageQuality:
    width: int
    height: int
    blur_score: float
    contrast_score: float
    issues: tuple[str, ...]


@dataclass(frozen=True)
class PreparedImage:
    data: object
    scale_x: float
    scale_y: float
    variant: str
    quality: ImageQuality


class OpenCvPreprocessor:
    """Assess a document and prepare a conservative OCR-friendly variant."""

    def __init__(self, *, minimum_width: int = 1200) -> None:
        self.minimum_width = minimum_width

    def prepare(self, image_path: Path) -> PreparedImage:
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "OpenCV is not installed. Install the 'ocr' project extra."
            ) from exc

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not decode image: {image_path}")

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast_score = float(gray.std())
        issues: list[str] = []
        if width < 800:
            issues.append(
                f"low resolution ({width}x{height}); small identifiers require review"
            )
        if blur_score < 75:
            issues.append(f"possible blur (score {blur_score:.1f})")
        if contrast_score < 32:
            issues.append(f"low contrast (score {contrast_score:.1f})")

        scale = max(1.0, self.minimum_width / width)
        if scale > 1:
            image = cv2.resize(
                image,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(
            lightness
        )
        enhanced = cv2.cvtColor(
            cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR
        )
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        enhanced = cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)

        quality = ImageQuality(
            width=width,
            height=height,
            blur_score=blur_score,
            contrast_score=contrast_score,
            issues=tuple(issues),
        )
        return PreparedImage(
            data=enhanced,
            scale_x=enhanced.shape[1] / width,
            scale_y=enhanced.shape[0] / height,
            variant="clahe_unsharp",
            quality=quality,
        )
