from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

import requests

from .evidence import (
    CompactExtractionClaims,
    EvidenceDocument,
    ExtractionDraft,
    claims_to_draft,
)
from .verification import (
    apply_focused_repair,
    merge_trusted_debtor_claims,
    merge_trusted_item_claims,
    merge_trusted_total_claims,
    normalize_delivery_presence,
    normalize_optional_placeholders,
    normalize_proven_ambiguities,
    relevant_evidence_payload,
    sanitize_unverified_claims,
    trusted_address_claims,
    trusted_company_claims,
    trusted_contact_name_claims,
    trusted_item_claims,
    trusted_total_claims,
    verify_claims,
)


_COMPACT_FEW_SHOT_MESSAGES = (
    {
        "role": "user",
        "content": (
            "Example: OCR spans e1='Contact: Ana Silva', e2='Billing Address', "
            "e3='Calle Mayor 8', e4='28013 Madrid', e5='Spain', "
            "e6='Delivery Address', e7='Avenida Sur 4', e8='41001 Seville', "
            "e9='Spain', e10='Status: PAID', e11='Payment date: 2026-04-09'."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "item_count": 0,
                "delivery_address_present": True,
                "claims": [
                    {
                        "path": "debtor.first_name",
                        "value": "Ana",
                        "evidence_ids": ["e1"],
                        "ambiguity": None,
                    },
                    {
                        "path": "debtor.last_name",
                        "value": "Silva",
                        "evidence_ids": ["e1"],
                        "ambiguity": None,
                    },
                    {
                        "path": "debtor.billing_address.street",
                        "value": "Calle Mayor 8",
                        "evidence_ids": ["e3"],
                        "ambiguity": None,
                    },
                    {
                        "path": "debtor.billing_address.zip",
                        "value": "28013",
                        "evidence_ids": ["e4"],
                        "ambiguity": None,
                    },
                    {
                        "path": "debtor.billing_address.city",
                        "value": "Madrid",
                        "evidence_ids": ["e4"],
                        "ambiguity": None,
                    },
                    {
                        "path": "debtor.billing_address.country",
                        "value": "Spain",
                        "evidence_ids": ["e5"],
                        "ambiguity": None,
                    },
                    {
                        "path": "debtor.delivery_address.street",
                        "value": "Avenida Sur 4",
                        "evidence_ids": ["e7"],
                        "ambiguity": None,
                    },
                    {
                        "path": "debtor.delivery_address.zip",
                        "value": "41001",
                        "evidence_ids": ["e8"],
                        "ambiguity": None,
                    },
                    {
                        "path": "debtor.delivery_address.city",
                        "value": "Seville",
                        "evidence_ids": ["e8"],
                        "ambiguity": None,
                    },
                    {
                        "path": "debtor.delivery_address.country",
                        "value": "Spain",
                        "evidence_ids": ["e9"],
                        "ambiguity": None,
                    },
                    {
                        "path": "payment.status",
                        "value": "PAID",
                        "evidence_ids": ["e10"],
                        "ambiguity": None,
                    },
                    {
                        "path": "payment.payment_date",
                        "value": "2026-04-09",
                        "evidence_ids": ["e11"],
                        "ambiguity": None,
                    },
                ],
            },
            separators=(",", ":"),
        ),
    },
    {
        "role": "user",
        "content": (
            "Example: OCR span e1='Status: OVERDUE', e2='Due date: 2026-05-01'. "
            "Preserve the source payment state and its labeled date."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "item_count": 0,
                "delivery_address_present": False,
                "claims": [
                    {
                        "path": "payment.status",
                        "value": "OVERDUE",
                        "evidence_ids": ["e1"],
                        "ambiguity": None,
                    },
                    {
                        "path": "payment.payment_date",
                        "value": "2026-05-01",
                        "evidence_ids": ["e2"],
                        "ambiguity": None,
                    },
                ],
            },
            separators=(",", ":"),
        ),
    },
)


class StructuredOrderParser(Protocol):
    uses_image: bool

    def parse(self, evidence: EvidenceDocument) -> ExtractionDraft: ...


@dataclass(frozen=True)
class CompatibleChatConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 90
    supports_json_schema: bool | None = None


class CompatibleChatOrderParser:
    """Schema-constrained parser for OpenAI-compatible chat endpoints.

    The adapter is provider-neutral: the endpoint and model are supplied by the
    operator. Financial validation still happens locally after parsing.
    """

    def __init__(self, config: CompatibleChatConfig) -> None:
        self.config = config
        self.uses_image = False
        self._strict_schema_supported = _supports_strict_json_schema(config)
        if _is_vision_model(config.model):
            raise ValueError(
                f"{config.model!r} is a vision model; configure a text-only model "
                "for the local OCR pipeline"
            )

    def parse(self, evidence: EvidenceDocument) -> ExtractionDraft:
        delivery_present, trusted_addresses = trusted_address_claims(evidence)
        trusted_debtor = [
            *trusted_addresses,
            *trusted_company_claims(evidence),
            *trusted_contact_name_claims(evidence),
        ]
        trusted_count, trusted_claims = trusted_item_claims(evidence)
        trusted_totals = trusted_total_claims(evidence)
        resolved_paths = sorted(
            claim.path
            for claim in [*trusted_debtor, *trusted_claims, *trusted_totals]
        )
        prompt = self._extraction_prompt(evidence, trusted_count, resolved_paths)
        compact = self._request_compact(prompt)
        compact = normalize_optional_placeholders(compact)
        compact = normalize_delivery_presence(compact, evidence)
        compact = normalize_proven_ambiguities(compact, evidence)
        compact = merge_trusted_debtor_claims(
            compact, delivery_present, trusted_debtor
        )
        compact = merge_trusted_item_claims(compact, trusted_count, trusted_claims)
        compact = merge_trusted_total_claims(compact, trusted_totals)
        issues = verify_claims(compact, evidence)
        if issues:
            repair_prompt = (
                "Repair only the rejected claims listed below. Return the complete compact JSON "
                "object, but do not change any unrelated claim, item_count, or delivery flag. "
                "Use only the focused OCR evidence. If a rejected value cannot be proven, return "
                "that claim with value null and a precise ambiguity.\n\n"
                f"Rejected claims:\n{json.dumps([issue.prompt_payload() for issue in issues], ensure_ascii=False)}\n\n"
                f"Previous object:\n{compact.model_dump_json()}\n\n"
                "Focused OCR evidence:\n"
                f"{json.dumps(relevant_evidence_payload(evidence, issues), ensure_ascii=False)}"
            )
            repaired = normalize_optional_placeholders(self._request_compact(repair_prompt))
            repaired = normalize_delivery_presence(repaired, evidence)
            repaired = normalize_proven_ambiguities(repaired, evidence)
            compact = apply_focused_repair(compact, repaired, issues)
            compact = merge_trusted_debtor_claims(
                compact, delivery_present, trusted_debtor
            )
            compact = merge_trusted_item_claims(compact, trusted_count, trusted_claims)
            compact = merge_trusted_total_claims(compact, trusted_totals)
            issues = verify_claims(compact, evidence)
        if issues:
            compact = sanitize_unverified_claims(compact, issues)
        return claims_to_draft(compact)

    def _extraction_prompt(
        self,
        evidence: EvidenceDocument,
        trusted_count: int | None,
        resolved_paths: list[str],
    ) -> str:
        table_note = (
            f"Local table analysis fixed item_count={trusted_count}."
            if trusted_count is not None
            else "No trustworthy local item table was found; infer item rows from OCR reading order."
        )
        resolved_note = (
            " Local deterministic analysis already resolved these printed claims: "
            f"{json.dumps(resolved_paths)}. Omit those paths and extract only unresolved claims."
            if resolved_paths
            else ""
        )
        local_note = table_note + resolved_note
        return (
            "Extract source facts from this purchase-order OCR evidence into compact field claims. "
            "Each claim has path, value, evidence_ids, and ambiguity. Cite only supplied OCR IDs; "
            "never invent content. Preserve Unicode, spelling, identifiers, payment states, and "
            "labeled dates. Extraction must not apply accounting policy or recompute totals. "
            "Payment status may be PAID, UNPAID, OVERDUE, PARTIALLY PAID, or REFUNDED; preserve "
            "exact semantics. Include a printed payment-related date under payment.payment_date. "
            "Set delivery_address_present when a delivery address is explicitly printed, even if "
            "its values equal billing. Split an unambiguous personal contact into first and last "
            "name; both may cite the same span. Split postal code and city only when the source "
            "format is clear; both may cite one span. For uncertainty, use null plus ambiguity. "
            "Treat '-' or N/A as absent only for optional fields, never required identity, address, "
            "item, or total fields. Numbers may use grouping separators, decimal quantities, zero "
            "VAT, and discounts through 100 percent. Paths are limited to order_date, "
            "external_reference, currency; debtor identity; billing/delivery address street, zip, "
            "city, country, email, telephone, additional_name, address_specification, district; "
            "payment.method/status/payment_date; items[N].sku/description/quantity/unit_net_price/"
            "vat_percent/discount_percent/source_total; totals.total_net/total_vat/total_gross. "
            f"{local_note}\n\nOCR evidence:\n"
            f"{json.dumps(evidence.prompt_payload(), ensure_ascii=False)}"
        )

    def _request_compact(self, prompt: str) -> CompactExtractionClaims:
        schema = CompactExtractionClaims.model_json_schema()
        strict_schema = _strict_json_schema(schema)
        strict = self._strict_schema_supported
        if not strict:
            prompt = f"JSON schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n{prompt}"
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "response_format": (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "purchase_order_extraction_draft",
                        "strict": True,
                        "schema": strict_schema,
                    },
                }
                if strict
                else {"type": "json_object"}
            ),
            "messages": [
                {
                    "role": "system",
                    "content": "You extract accounting documents into strict JSON.",
                },
                *_COMPACT_FEW_SHOT_MESSAGES,
                {"role": "user", "content": prompt},
            ],
        }
        if "api.groq.com" in self.config.base_url.casefold() and self.config.model.startswith(
            "openai/gpt-oss-"
        ):
            payload["reasoning_format"] = "hidden"
            payload["reasoning_effort"] = "low"
        response = self._post(payload)
        if not response.ok:
            detail = _http_error_detail(response)
            if strict and _is_schema_rejection(detail):
                # OpenAI-compatible gateways sometimes accept json_schema in a
                # capability probe but reject nullable/full production schemas.
                # Retry once in JSON mode; local validation remains mandatory.
                payload["response_format"] = {"type": "json_object"}
                self._strict_schema_supported = False
                payload["messages"][-1] = {
                    "role": "user",
                    "content": (
                        f"JSON schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
                        f"{prompt}"
                    ),
                }
                response = self._post(payload)
                detail = _http_error_detail(response) if not response.ok else ""
            if not response.ok:
                raise RuntimeError(
                    f"LLM extraction request failed: HTTP {response.status_code}: {detail}"
                )
        try:
            body = response.json()
        except requests.JSONDecodeError as exc:
            raise RuntimeError("LLM extraction response was not valid JSON") from exc

        content = body["choices"][0]["message"]["content"]
        parsed = _parse_json_object(content)
        return CompactExtractionClaims.model_validate(parsed)

    def _post(self, payload: dict) -> requests.Response:
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        try:
            return requests.post(
                endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Accept": "application/json",
                    "User-Agent": "fakturama-image-to-cash/0.1",
                },
                timeout=self.config.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"LLM extraction request failed: {exc}") from exc


def _is_vision_model(model: str) -> bool:
    normalized = model.casefold()
    return any(
        marker in normalized
        for marker in ("qwen3.6", "vision", "llava", "scout", "maverick")
    )


def _supports_strict_json_schema(config: CompatibleChatConfig) -> bool:
    """Resolve endpoint capability conservatively; explicit configuration wins."""
    if config.supports_json_schema is not None:
        return config.supports_json_schema
    return (
        "api.groq.com" in config.base_url.casefold()
        and config.model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
    )


def _parse_json_object(content: str) -> dict:
    content = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    if fenced:
        content = fenced.group(1)
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        value = _first_embedded_json_object(content)
    if not isinstance(value, dict):
        raise ValueError("LLM response must be one JSON object")
    return value


def _first_embedded_json_object(content: str) -> dict:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            value, _ = decoder.raw_decode(content[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("LLM response was not valid JSON")


def _http_error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
        error = payload.get("error", payload)
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error)[:500]
        return str(error)[:500]
    except (requests.JSONDecodeError, AttributeError):
        return response.text.strip()[:500] or response.reason


def _is_schema_rejection(detail: str) -> bool:
    normalized = detail.casefold()
    return "schema" in normalized and any(
        marker in normalized
        for marker in ("invalid", "response_format", "must have", "unsupported")
    )


def _strict_json_schema(value):
    """Make Pydantic JSON Schema compatible with Groq constrained decoding."""
    if isinstance(value, list):
        return [_strict_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    transformed = {
        key: _strict_json_schema(item)
        for key, item in value.items()
        if key != "default"
    }
    properties = transformed.get("properties")
    if isinstance(properties, dict):
        transformed["required"] = list(properties)
        transformed["additionalProperties"] = False
    return transformed
