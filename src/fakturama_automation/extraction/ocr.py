from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Protocol

from .preprocessing import ImageQuality, OpenCvPreprocessor


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float
    bounding_box: tuple[int, int, int, int]
    variant: str = "original"
    page: int = 1
    model: str = "unknown"


@dataclass(frozen=True)
class LayoutRegion:
    kind: str
    bounding_box: tuple[int, int, int, int]
    text: str = ""


@dataclass(frozen=True)
class OcrResult:
    lines: list[OcrLine]
    regions: tuple[LayoutRegion, ...] = ()
    quality: ImageQuality | None = None

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    def prompt_payload(self) -> dict:
        return {
            "quality": {
                "width": self.quality.width,
                "height": self.quality.height,
                "issues": list(self.quality.issues),
            }
            if self.quality
            else None,
            "regions": [
                {
                    "kind": region.kind,
                    "bounding_box": region.bounding_box,
                    "text": region.text,
                }
                for region in self.regions
            ],
            "lines": [
                {
                    "text": line.text,
                    "confidence": round(line.confidence, 4),
                    "bounding_box": line.bounding_box,
                    "variant": line.variant,
                }
                for line in self.lines
            ],
        }


class OcrBackend(Protocol):
    def recognize(self, image_path: Path) -> OcrResult: ...


class EasyOcrBackend:
    """Lazy EasyOCR adapter so validation/simulation do not load ML libraries."""

    def __init__(self, languages: tuple[str, ...] = ("en",), gpu: bool = False) -> None:
        self.languages = languages
        self.gpu = gpu
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:  # pragma: no cover - depends on optional package
                raise RuntimeError(
                    "EasyOCR is not installed. Install the 'ocr' project extra."
                ) from exc
            self._reader = easyocr.Reader(list(self.languages), gpu=self.gpu)
        return self._reader

    def recognize(self, image_path: Path) -> OcrResult:
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        raw = self._get_reader().readtext(str(image_path), detail=1, paragraph=False)
        lines: list[OcrLine] = []
        for polygon, text, confidence in raw:
            xs = [int(point[0]) for point in polygon]
            ys = [int(point[1]) for point in polygon]
            lines.append(
                OcrLine(
                    text=str(text).strip(),
                    confidence=float(confidence),
                    bounding_box=(min(xs), min(ys), max(xs), max(ys)),
                    model="easyocr:" + ",".join(self.languages),
                )
            )
        return OcrResult(lines=lines)


class PaddleOcrBackend:
    """Local PaddleOCR adapter with optional PP-StructureV3 layout analysis.

    MKLDNN is disabled deliberately because the current Windows Paddle runtime
    can fail while converting PIR attributes through oneDNN.
    """

    def __init__(
        self,
        language: str = "en",
        *,
        gpu: bool = False,
        layout_analysis: bool = False,
        cache_directory: Path = Path(".cache/paddlex"),
        preprocessor: OpenCvPreprocessor | None = None,
    ) -> None:
        self.language = _paddle_language(language)
        self.device = "gpu" if gpu else "cpu"
        self.layout_analysis = layout_analysis
        self.cache_directory = cache_directory.resolve()
        self.preprocessor = preprocessor or OpenCvPreprocessor()
        self._ocr = None
        self._structure = None

    def _configure_runtime(self) -> None:
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(self.cache_directory))
        os.environ.setdefault("FLAGS_use_mkldnn", "0")

    def _get_ocr(self):
        if self._ocr is None:
            self._configure_runtime()
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "PaddleOCR is not installed. Install the 'ocr' project extra."
                ) from exc
            self._ocr = PaddleOCR(
                lang=self.language,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
                device=self.device,
            )
        return self._ocr

    def _get_structure(self):
        if self._structure is None:
            self._configure_runtime()
            try:
                from paddleocr import PPStructureV3
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "PP-StructureV3 is not installed. Install the 'ocr' project extra."
                ) from exc
            self._structure = PPStructureV3(
                lang=self.language,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                use_seal_recognition=False,
                use_formula_recognition=False,
                use_chart_recognition=False,
                use_table_recognition=True,
                enable_mkldnn=False,
                device=self.device,
            )
        return self._structure

    def recognize(self, image_path: Path) -> OcrResult:
        prepared = self.preprocessor.prepare(image_path)
        predictions = list(
            self._get_ocr().predict(
                prepared.data,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        )
        lines: list[OcrLine] = []
        for prediction in predictions:
            result = prediction.json.get("res", prediction.json)
            texts = result.get("rec_texts", [])
            scores = result.get("rec_scores", [])
            boxes = result.get("rec_boxes", [])
            for text, confidence, box in zip(texts, scores, boxes, strict=False):
                x1, y1, x2, y2 = (int(value) for value in box)
                lines.append(
                    OcrLine(
                        text=str(text).strip(),
                        confidence=float(confidence),
                        bounding_box=(
                            round(x1 / prepared.scale_x),
                            round(y1 / prepared.scale_y),
                            round(x2 / prepared.scale_x),
                            round(y2 / prepared.scale_y),
                        ),
                        variant=prepared.variant,
                        model="PP-OCRv6_medium",
                    )
                )
        regions = self._recognize_regions(image_path) if self.layout_analysis else ()
        return OcrResult(lines=lines, regions=regions, quality=prepared.quality)

    def _recognize_regions(self, image_path: Path) -> tuple[LayoutRegion, ...]:
        predictions = list(
            self._get_structure().predict(
                str(image_path),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                use_seal_recognition=False,
                use_formula_recognition=False,
                use_chart_recognition=False,
                use_table_recognition=True,
            )
        )
        regions: list[LayoutRegion] = []
        for prediction in predictions:
            result = prediction.json.get("res", prediction.json)
            for block in result.get("parsing_res_list", []):
                box = block.get("block_bbox") or block.get("bbox")
                if box is None or len(box) != 4:
                    continue
                regions.append(
                    LayoutRegion(
                        kind=str(block.get("block_label") or block.get("label") or "text"),
                        bounding_box=tuple(int(value) for value in box),
                        text=str(
                            block.get("block_content")
                            or block.get("block_text")
                            or block.get("content")
                            or ""
                        ),
                    )
                )
        return tuple(regions)


def _paddle_language(language: str) -> str:
    mapping = {"de": "german", "ger": "german", "en": "en"}
    return mapping.get(language.casefold(), language)
