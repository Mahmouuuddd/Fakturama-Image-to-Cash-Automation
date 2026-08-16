from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from fakturama_automation.domain.errors import ManualReviewRequired
from fakturama_automation.domain.validation import (
    Severity,
    ValidationIssue,
    ValidationReport,
    validate_order,
)
from fakturama_automation.extraction.ocr import EasyOcrBackend, PaddleOcrBackend
from fakturama_automation.extraction.evidence import build_evidence_document
from fakturama_automation.extraction.parser import (
    CompatibleChatConfig,
    CompatibleChatOrderParser,
)
from fakturama_automation.extraction.pipeline import (
    ExtractionOutcome,
    ImageOrderExtractor,
    JsonOrderExtractor,
)
from fakturama_automation.extraction.rules import SpatialOrderParser
from fakturama_automation.gateways.simulated import SimulatedFakturamaGateway
from fakturama_automation.infrastructure.checkpoints import CheckpointStore
from fakturama_automation.infrastructure.evidence import EvidenceRecorder
from fakturama_automation.infrastructure.review import write_review_packet
from fakturama_automation.workflow.engine import WorkflowRunner, WorkflowState


DEFAULT_EXE = Path(r"C:\Program Files\Fakturama2\Fakturama.exe")
DEFAULT_PROFILE = Path("config/fakturama-2.2.0-en.json")


@dataclass(frozen=True)
class RunContext:
    workflow_id: str
    directory: Path
    recorder: EvidenceRecorder
    checkpoint: CheckpointStore


def load_environment(env_file: Path | None = None) -> bool:
    """Load .env without overriding variables supplied by the operating system."""
    configured_path = env_file or Path(os.getenv("FAKTURAMA_ENV_FILE", ".env"))
    return load_dotenv(dotenv_path=configured_path, override=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fakturama-automation",
        description="Verified Fakturama image-to-Order-to-Invoice automation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run", help="normal end-to-end image-to-Fakturama workflow"
    )
    _add_extraction_arguments(run)
    run.add_argument("--executable", type=Path, default=DEFAULT_EXE)
    run.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    run.add_argument("--runs-dir", type=Path, default=Path("runs"))
    run.add_argument(
        "--confirm-writes",
        action="store_true",
        help="required acknowledgement that Fakturama records will be created",
    )

    extract = subparsers.add_parser(
        "extract", help="developer diagnostic: OCR and structure one image"
    )
    _add_extraction_arguments(extract)
    extract.add_argument("--output", type=Path)
    extract.add_argument("--runs-dir", type=Path, default=Path("runs"))

    validate = subparsers.add_parser(
        "validate", help="developer diagnostic: validate extracted Order JSON"
    )
    validate.add_argument("order_json", type=Path)

    simulate = subparsers.add_parser(
        "simulate", help="developer diagnostic: run against the in-memory gateway"
    )
    simulate.add_argument("order_json", type=Path)
    simulate.add_argument("--runs-dir", type=Path, default=Path("runs"))

    inspect = subparsers.add_parser(
        "inspect-uia", help="developer diagnostic: export the accessible control tree"
    )
    inspect.add_argument("--executable", type=Path, default=DEFAULT_EXE)
    inspect.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    inspect.add_argument("--output", type=Path, default=Path("runs/uia-tree.json"))
    return parser


def _add_extraction_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--languages",
        default=os.getenv("OCR_LANGUAGES", "en"),
        help="configured local OCR languages; Paddle uses the first configured model",
    )
    parser.add_argument(
        "--ocr-backend",
        choices=("paddle", "easyocr"),
        default=os.getenv("OCR_BACKEND", "paddle"),
        help="local OCR engine (default: paddle)",
    )
    parser.add_argument(
        "--parser",
        choices=("llm", "auto", "rules"),
        default="llm",
        help=(
            "text-only LLM is the trusted default; auto is a compatibility alias; "
            "rules is a legacy extraction diagnostic only"
        ),
    )
    parser.add_argument(
        "--layout-analysis",
        action="store_true",
        help="optional PP-StructureV3 diagnostics (slow on CPU; not required)",
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL"))
    parser.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY"))
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL"))
    parser.add_argument(
        "--llm-supports-json-schema",
        type=_optional_bool,
        default=_optional_bool(os.getenv("LLM_SUPPORTS_JSON_SCHEMA", "auto")),
        metavar="auto|true|false",
        help="explicit endpoint capability; auto recognizes documented Groq models",
    )


def main(argv: list[str] | None = None) -> int:
    load_environment()
    args = build_parser().parse_args(argv)
    context: RunContext | None = None
    outcome: ExtractionOutcome | None = None
    try:
        if args.command == "validate":
            order, report = JsonOrderExtractor().extract(args.order_json)
            _print_validation(order, report)
            return 0 if report.valid else 2

        if args.command == "simulate":
            order, report = JsonOrderExtractor().extract(args.order_json)
            _print_validation(order, report)
            report.raise_for_errors()
            result = _run_workflow(order, SimulatedFakturamaGateway(), args.runs_dir)
            _print_result(result)
            return 0

        if args.command == "inspect-uia":
            from fakturama_automation.gateways.uia import inspect_uia_tree

            count = inspect_uia_tree(args.executable, args.profile, args.output)
            print(f"Exported {count} UIA elements to {args.output.resolve()}")
            return 0

        context = _create_run_context(args.runs_dir, args.image)
        _save_checkpoint(
            context,
            WorkflowState.CREATED,
            source_image=str(args.image.resolve()),
        )
        context.recorder.record(
            "RUN_CREATED",
            workflow_id=context.workflow_id,
            source_image=str(args.image.resolve()),
            command=args.command,
        )

        if args.command == "extract" and args.parser == "rules":
            return _run_legacy_extraction(args, context)

        if args.command == "run" and args.parser == "rules":
            raise ValueError(
                "the legacy coordinate parser is diagnostic-only and cannot authorize UI writes"
            )

        if args.command == "run" and not args.confirm_writes:
            raise ValueError(
                "refusing real UI writes without --confirm-writes; no Fakturama changes were made"
            )

        outcome = _extract_image(
            args,
            evidence_callback=lambda evidence: _persist_ocr_stage(context, evidence),
        )
        _persist_extraction(context, outcome)
        _print_validation(outcome.order, outcome.report, outcome.draft)

        diagnostic_output: Path | None = None
        if args.command == "extract" and outcome.order is not None:
            diagnostic_output = args.output or context.directory / "order.json"
            diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
            diagnostic_output.write_text(
                outcome.order.model_dump_json(indent=2), encoding="utf-8"
            )
            print(f"Structured order written to {diagnostic_output.resolve()}")

        if outcome.order is None or outcome.report.requires_review:
            packet = write_review_packet(
                args.image,
                outcome.order,
                outcome.report,
                context.directory / "review.json",
                draft=outcome.draft,
                evidence_document=outcome.evidence,
                workflow_id=context.workflow_id,
            )
            _save_checkpoint(
                context,
                WorkflowState.MANUAL_REVIEW,
                review_packet=str(packet.resolve()),
                issue_codes=[issue.code for issue in outcome.report.issues],
            )
            context.recorder.record(
                "MANUAL_REVIEW",
                review_packet=str(packet.resolve()),
                issue_codes=[issue.code for issue in outcome.report.issues],
            )
            print(f"Human review required; packet written to {packet.resolve()}")
            return 3

        if args.command == "extract":
            return 0

        from fakturama_automation.gateways.uia import PywinautoFakturamaGateway

        gateway = PywinautoFakturamaGateway(args.executable, args.profile)
        result = _run_workflow(
            outcome.order,
            gateway,
            args.runs_dir,
            context=context,
            initial_state=WorkflowState.EXTRACTED,
        )
        _print_result(result)
        return 0
    except ManualReviewRequired as exc:
        packet = None
        if context is not None and outcome is not None:
            checkpoint = context.checkpoint.load() or {}
            checkpoint_details = checkpoint.get("details") or {}
            workflow_state = str(
                checkpoint_details.get("prior_state")
                or checkpoint.get("state", "MANUAL_REVIEW")
            )
            review_report = ValidationReport(
                issues=[
                    ValidationIssue(
                        code="WORKFLOW_AMBIGUITY",
                        path="workflow",
                        message=str(exc),
                    )
                ]
            )
            packet = write_review_packet(
                args.image,
                outcome.order,
                review_report,
                context.directory / "review.json",
                draft=outcome.draft,
                evidence_document=outcome.evidence,
                workflow_id=context.workflow_id,
                workflow_state=workflow_state,
            )
            context.recorder.record(
                "WORKFLOW_REVIEW_PACKET", review_packet=str(packet.resolve())
            )
        print(f"Manual review required: {exc}", file=sys.stderr)
        if packet is not None:
            print(f"Review packet written to {packet.resolve()}", file=sys.stderr)
        return 3
    except Exception as exc:
        if context is not None:
            context.recorder.record(
                "FAILED", error=str(exc), error_type=type(exc).__name__
            )
            _save_checkpoint(
                context,
                WorkflowState.FAILED,
                error=str(exc),
                error_type=type(exc).__name__,
            )
        print(f"Automation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _extract_image(args, *, evidence_callback=None) -> ExtractionOutcome:
    ocr = _build_ocr(args)
    values = (args.llm_base_url, args.llm_api_key, args.llm_model)
    if not all(values):
        raise ValueError(
            "the default text-only parser requires LLM_BASE_URL, LLM_API_KEY, and "
            "LLM_MODEL in .env (or equivalent command options)"
        )
    parser = CompatibleChatOrderParser(
        CompatibleChatConfig(
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            model=args.llm_model,
            supports_json_schema=args.llm_supports_json_schema,
        )
    )
    return ImageOrderExtractor(ocr, parser).extract(
        args.image, evidence_callback=evidence_callback
    )


def _build_ocr(args):
    languages = tuple(
        item.strip() for item in args.languages.split(",") if item.strip()
    ) or ("en",)
    if args.ocr_backend == "paddle":
        return PaddleOcrBackend(
            languages[0],
            gpu=args.gpu,
            layout_analysis=args.layout_analysis,
        )
    return EasyOcrBackend(languages, gpu=args.gpu)


def _optional_bool(value: str | None) -> bool | None:
    normalized = str(value or "auto").strip().casefold()
    if normalized == "auto":
        return None
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected auto, true, or false")


def _run_legacy_extraction(args, context: RunContext) -> int:
    """Preserve the old coordinate parser without allowing it into the trusted run path."""
    ocr_result = _build_ocr(args).recognize(args.image)
    _persist_ocr_stage(context, build_evidence_document(args.image, ocr_result))
    order = SpatialOrderParser().parse(ocr_result, image_path=args.image)
    report = validate_order(order, require_evidence=True)
    report.issues.append(
        ValidationIssue(
            code="LEGACY_COORDINATE_PARSER",
            path="parser",
            message=(
                "supplier/coordinate parsing is retained for diagnostics and cannot authorize "
                "Fakturama writes"
            ),
            severity=Severity.WARNING,
        )
    )
    output = args.output or context.directory / "legacy-order.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(order.model_dump_json(indent=2), encoding="utf-8")
    _write_json(context.directory / "validation.json", _validation_payload(report))
    _print_validation(order, report)
    packet = write_review_packet(
        args.image,
        order,
        report,
        context.directory / "review.json",
        workflow_id=context.workflow_id,
    )
    _save_checkpoint(
        context,
        WorkflowState.MANUAL_REVIEW,
        review_packet=str(packet.resolve()),
        legacy_parser=True,
    )
    print(f"Legacy diagnostic order written to {output.resolve()}")
    print(f"Human review packet written to {packet.resolve()}")
    return 3


def _create_run_context(runs_dir: Path, source: Path) -> RunContext:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    source_label = re.sub(r"[^A-Za-z0-9_-]", "_", source.stem)[:32] or "document"
    workflow_id = f"{timestamp}-{source_label}-{uuid4().hex[:8]}"
    directory = runs_dir / workflow_id
    recorder = EvidenceRecorder(directory)
    return RunContext(
        workflow_id=workflow_id,
        directory=directory,
        recorder=recorder,
        checkpoint=CheckpointStore(directory / "checkpoint.json"),
    )


def _persist_extraction(context: RunContext, outcome: ExtractionOutcome) -> None:
    _write_json(
        context.directory / "extraction.json",
        {
            "workflow_id": context.workflow_id,
            "draft": outcome.draft.model_dump(mode="json"),
            "order": outcome.order.model_dump(mode="json") if outcome.order else None,
        },
    )
    _write_json(
        context.directory / "evidence.json", outcome.evidence.model_dump(mode="json")
    )
    _write_json(
        context.directory / "validation.json", _validation_payload(outcome.report)
    )
    context.recorder.record(
        "EXTRACTION_COMPLETED", order_authorized=outcome.order is not None
    )
    context.recorder.record(
        "VALIDATION_COMPLETED",
        valid=outcome.report.valid,
        requires_review=outcome.report.requires_review,
        issue_codes=[issue.code for issue in outcome.report.issues],
    )
    _save_checkpoint(
        context,
        WorkflowState.EXTRACTED,
        valid=outcome.report.valid,
        requires_review=outcome.report.requires_review,
        external_reference=(
            outcome.order.external_reference
            if outcome.order
            else outcome.draft.external_reference.value
        ),
    )


def _persist_ocr_stage(context: RunContext, evidence) -> None:
    _write_json(context.directory / "evidence.json", evidence.model_dump(mode="json"))
    context.recorder.record(
        "OCR_COMPLETED",
        evidence_spans=len(evidence.spans),
        source_sha256=evidence.source_sha256,
    )
    _save_checkpoint(
        context,
        WorkflowState.OCR_COMPLETED,
        evidence_file=str((context.directory / "evidence.json").resolve()),
        evidence_spans=len(evidence.spans),
        source_sha256=evidence.source_sha256,
    )


def _save_checkpoint(context: RunContext, state: WorkflowState, **details) -> None:
    context.checkpoint.save(
        {
            "workflow_id": context.workflow_id,
            "state": state.value,
            "details": details,
        }
    )


def _validation_payload(report) -> dict:
    return {
        "valid": report.valid,
        "requires_review": report.requires_review,
        "issues": [
            {
                "code": issue.code,
                "path": issue.path,
                "severity": issue.severity.value,
                "message": issue.message,
            }
            for issue in report.issues
        ],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _run_workflow(
    order,
    gateway,
    runs_dir: Path,
    *,
    context: RunContext | None = None,
    initial_state: WorkflowState = WorkflowState.CREATED,
):
    if context is None:
        context = _create_run_context(runs_dir, Path(order.external_reference))
        context.recorder.record(
            "RUN_CREATED",
            workflow_id=context.workflow_id,
            external_reference=order.external_reference,
            order=order.model_dump(mode="json", exclude={"evidence"}),
        )
    runner = WorkflowRunner(
        gateway,
        context.checkpoint,
        context.recorder,
        workflow_id=context.workflow_id,
        initial_state=initial_state,
    )
    return runner.run(order)


def _print_validation(order, report, draft=None) -> None:
    external_reference = (
        order.external_reference
        if order is not None
        else (draft.external_reference.value if draft is not None else None)
    )
    item_count = len(order.items) if order is not None else len(draft.items) if draft else 0
    total = (
        str(order.totals.total_gross)
        if order is not None
        else draft.totals.total_gross.value if draft else None
    )
    currency = order.currency if order is not None else draft.currency.value if draft else None
    print(
        json.dumps(
            {
                "external_reference": external_reference,
                "items": item_count,
                "total": total,
                "currency": currency,
                **_validation_payload(report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _print_result(result) -> None:
    print(
        json.dumps(
            {
                "state": result.state.value,
                "order_number": result.order_document.number,
                "invoice_number": result.invoice_document.number,
                "transaction_id": result.order_document.transaction_id,
                "run_directory": str(result.run_directory.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
