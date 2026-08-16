from __future__ import annotations

import json
import re
from pathlib import Path

from fakturama_automation.domain.models import FieldEvidence, OrderInput
from fakturama_automation.domain.validation import ValidationReport
from fakturama_automation.extraction.evidence import (
    EvidenceDocument,
    ExtractedField,
    ExtractionDraft,
)


def write_review_packet(
    image_path: Path,
    order: OrderInput | None,
    report: ValidationReport,
    packet_path: Path,
    *,
    draft: ExtractionDraft | None = None,
    evidence_document: EvidenceDocument | None = None,
    workflow_id: str | None = None,
    workflow_state: str = "MANUAL_REVIEW",
) -> Path:
    """Persist the uncertain fields and source crops for a human reviewer."""
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    assets_directory = packet_path.parent / f"{packet_path.stem}-assets"
    crops: dict[str, str] = {}

    field_evidence: dict[str, FieldEvidence] = dict(order.evidence) if order else {}
    if draft is not None and evidence_document is not None:
        span_index = evidence_document.span_index()
        for issue in report.issues:
            extracted = _draft_field_at_path(draft, issue.path)
            if extracted is None:
                continue
            spans = [span_index[item] for item in extracted.evidence_ids if item in span_index]
            if not spans:
                continue
            pages = {span.page for span in spans}
            box = None
            if len(pages) == 1:
                box = (
                    min(span.bbox[0] for span in spans),
                    min(span.bbox[1] for span in spans),
                    max(span.bbox[2] for span in spans),
                    max(span.bbox[3] for span in spans),
                )
            field_evidence.setdefault(
                issue.path,
                FieldEvidence(
                    source_text=" | ".join(span.text for span in spans),
                    confidence=min(span.confidence for span in spans),
                    bounding_box=box,
                    evidence_ids=extracted.evidence_ids,
                    page=spans[0].page,
                ),
            )

    evidence_paths = {issue.path for issue in report.issues if issue.path in field_evidence}
    if evidence_paths:
        try:
            from PIL import Image
        except ImportError:  # pragma: no cover - installed with OCR extra
            Image = None
        if Image is not None:
            assets_directory.mkdir(parents=True, exist_ok=True)
            with Image.open(image_path) as image:
                for field_path in sorted(evidence_paths):
                    box = field_evidence[field_path].bounding_box
                    if box is None:
                        continue
                    left, top, right, bottom = box
                    padding = max(8, round((bottom - top) * 1.5))
                    crop = image.crop(
                        (
                            max(0, left - padding),
                            max(0, top - padding),
                            min(image.width, right + padding),
                            min(image.height, bottom + padding),
                        )
                    )
                    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", field_path)
                    crop_path = assets_directory / f"{safe_name}.png"
                    crop.save(crop_path)
                    crops[field_path] = str(crop_path.resolve())

    saved_or_later = workflow_state in {
        "ORDER_SAVED",
        "ORDER_VERIFIED",
        "INVOICE_OPEN",
        "PAYMENT_APPLIED",
        "INVOICE_SAVED",
        "FINAL_VERIFIED",
    }
    instructions = [
        "Compare every listed field with its source crop or the original image.",
        "Correct the structured order JSON; do not edit calculated totals blindly.",
        "Run validation again before allowing Fakturama writes.",
    ]
    if saved_or_later:
        instructions = [
            "Do not rerun automatic record creation: the Order may already be saved.",
            "Inspect checkpoint.json and the Fakturama records using their persisted numbers.",
            "Resolve this state manually unless duplicate-safe resume is explicitly implemented.",
        ]
    packet = {
        "status": "HUMAN_REVIEW_REQUIRED",
        "workflow_state": workflow_state,
        "workflow_id": workflow_id,
        "source_image": str(image_path.resolve()),
        "issues": [
            {
                "path": issue.path,
                "code": issue.code,
                "severity": issue.severity.value,
                "message": issue.message,
                "source_crop": crops.get(issue.path),
            }
            for issue in report.issues
        ],
        "order": order.model_dump(mode="json") if order else None,
        "extraction_draft": draft.model_dump(mode="json") if draft else None,
        "review_instructions": instructions,
    }
    packet_path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return packet_path


def _draft_field_at_path(
    draft: ExtractionDraft, field_path: str
) -> ExtractedField | None:
    current = draft
    for part in re.findall(r"[^.\[\]]+", field_path):
        try:
            current = current[int(part)] if part.isdigit() else getattr(current, part)
        except (AttributeError, IndexError, TypeError):
            return None
    return current if isinstance(current, ExtractedField) else None
