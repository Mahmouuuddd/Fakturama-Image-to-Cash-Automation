import os
import json
from pathlib import Path
from types import SimpleNamespace

import fakturama_automation.cli as cli
from fakturama_automation.cli import load_environment
from fakturama_automation.domain.models import OrderInput
from fakturama_automation.domain.validation import ValidationReport
from fakturama_automation.gateways.simulated import SimulatedFakturamaGateway


def test_dotenv_is_loaded_without_overriding_existing_environment(
    tmp_path: Path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_BASE_URL=https://example.invalid/v1\n"
        "LLM_API_KEY=file-secret\n"
        "LLM_MODEL=file-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "process-secret")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    assert load_environment(env_file)
    assert os.environ["LLM_BASE_URL"] == "https://example.invalid/v1"
    assert os.environ["LLM_API_KEY"] == "process-secret"
    assert os.environ["LLM_MODEL"] == "file-model"


def test_one_run_command_owns_extraction_artifacts_and_workflow(
    tmp_path: Path, monkeypatch
) -> None:
    image = tmp_path / "order.png"
    image.write_bytes(b"test-image")
    order = OrderInput.model_validate_json(
        Path("examples/order.json").read_text(encoding="utf-8")
    )
    draft = SimpleNamespace(
        external_reference=SimpleNamespace(value=order.external_reference),
        items=order.items,
        totals=SimpleNamespace(
            total_gross=SimpleNamespace(value=str(order.totals.total_gross))
        ),
        currency=SimpleNamespace(value=order.currency),
        model_dump=lambda mode: {"test": "draft"},
    )
    evidence = SimpleNamespace(
        spans=(),
        source_sha256="test-hash",
        model_dump=lambda mode: {"test": "evidence"},
    )
    outcome = SimpleNamespace(
        order=order,
        draft=draft,
        evidence=evidence,
        report=ValidationReport(),
    )
    def fake_extract(args, *, evidence_callback=None):
        if evidence_callback is not None:
            evidence_callback(evidence)
        return outcome

    monkeypatch.setattr(cli, "_extract_image", fake_extract)
    import fakturama_automation.gateways.uia as uia

    monkeypatch.setattr(
        uia,
        "PywinautoFakturamaGateway",
        lambda executable, profile: SimulatedFakturamaGateway(),
    )

    result = cli.main(
        [
            "run",
            str(image),
            "--confirm-writes",
            "--runs-dir",
            str(tmp_path / "runs"),
        ]
    )

    assert result == 0
    run_directories = list((tmp_path / "runs").iterdir())
    assert len(run_directories) == 1
    run_directory = run_directories[0]
    assert (run_directory / "evidence.json").is_file()
    assert (run_directory / "extraction.json").is_file()
    assert (run_directory / "validation.json").is_file()
    checkpoint = json.loads((run_directory / "checkpoint.json").read_text("utf-8"))
    assert checkpoint["state"] == "FINAL_VERIFIED"
