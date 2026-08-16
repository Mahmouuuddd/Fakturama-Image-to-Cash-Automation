# Fakturama Image-to-Cash Automation

This project turns one purchase-order image into a verified Fakturama Order and
its linked Invoice. Images are processed locally. A text-only LLM interprets
locally produced OCR evidence, while deterministic code grounds every proposed
value, reconciles all accounting amounts, and decides whether desktop writes
are safe.

This README is the canonical repository documentation. Local planning,
architecture, workflow, and assignment Markdown files are intentionally kept
outside Git; the operational information required to install, configure, run,
test, and understand the submitted project is maintained here.

The system fails closed: any warning, ambiguity, financial contradiction,
unknown evidence ID, uncertain identity, UIA ambiguity, or failed read-back
stops the workflow for review.

## Normal operation: one command

After installation and `.env` setup, the complete operational workflow is:

```powershell
fakturama-automation run ".\order.jpg" --confirm-writes
```

That one command owns the complete sequence:

```text
image -> OpenCV -> PaddleOCR -> EvidenceDocument -> trusted local table claims
      -> text LLM semantic claims -> verify -> at most one focused repair
      -> local grounding -> OrderInput -> deterministic validation
      -> Fakturama Order -> linked Invoice -> persisted read-back
```

There are only three business outcomes:

- `SUCCESS` (exit `0`): Order and linked Invoice were saved and verified.
- `MANUAL_REVIEW` (exit `3`): durable artifacts were saved and no further
  writes are allowed.
- `FAILED` (exit `1`): a runtime or unsupported automation error stopped the
  workflow safely.

`--confirm-writes` is mandatory because `run` can create real records.

## Installation

Use Python 3.11 or newer on Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ocr,uia]"
```

PaddleOCR downloads models on first use and caches them under
`.cache/paddlex/official_models`. Messages saying cached model files already
exist are normal. The `No ccache found` warning is normally harmless for
prebuilt inference packages.

## Configuration

Create the local environment file:

```powershell
Copy-Item .env.example .env
notepad .env
```

Recommended Lightning AI configuration for large documents:

```dotenv
LLM_BASE_URL=https://lightning.ai/api/v1
LLM_API_KEY=your-lightning-secret-key
LLM_MODEL=openai/gpt-5.4-mini-2026-03-17
LLM_SUPPORTS_JSON_SCHEMA=false
OCR_BACKEND=paddle
OCR_LANGUAGES=en
```

This exact Lightning model was live-tested on 2026-08-16. JSON mode works, but
the gateway rejects the project's full nullable strict schema, so the explicit
capability is `false`. Local Pydantic and deterministic claim verification
remain mandatory. The adapter also falls back once to JSON mode if a provider
rejects a configured strict schema.

Groq remains supported with `https://api.groq.com/openai/v1`,
`openai/gpt-oss-120b`, and `LLM_SUPPORTS_JSON_SCHEMA=true`. Capability is
configured explicitly because an OpenAI-compatible endpoint does not imply
support for strict JSON Schema.

The `.env` file is ignored by Git. Never place a real key in source, examples,
tests, or audit artifacts.

The image itself is never sent to the LLM. OCR business text, bounding boxes,
confidence values, and local evidence IDs are sent, so the LLM call is not a
fully offline or zero-disclosure operation.

## Trust boundary

Locally inferred item-table cells are authoritative when their role and value
are unambiguous. They become item claims directly and are omitted from the LLM
task. The remote model handles unresolved cells and semantic fields such as
identity, address meaning, dates, currency, and payment state.

Clearly printed Net, VAT, and Gross totals are also claimed locally when one
exact label has one unambiguous amount directly below or beside it. This is a
relative OCR-box association, not a template coordinate or a calculated total.
Ambiguous total sections continue to use the semantic extraction/review path.

The remote model returns a compact allowlisted claims array. Every claim has a
field path, value, OCR evidence IDs, and optional ambiguity. Local code rejects
unknown or duplicate paths and verifies ID existence, OCR value support, and
item row/column ownership before expanding `ExtractionDraft`. On failure it may
make exactly one focused repair request containing only relevant evidence.
Unresolved claims become explicit review ambiguity; there is no retry loop.

Names, addresses, descriptions, references, and SKUs retain source Unicode and
spelling. The system does not translate, transliterate, fuzzy-correct, or strip
diacritics to force a match.

OCR evidence is sent in a compact columnar representation; repeated model,
variant, reading-order, and source-hash metadata remains local in
`evidence.json`. For Groq GPT-OSS, the compact claims use strict JSON Schema.
Other compatible text models use JSON Object Mode followed by local Pydantic
and claim-path validation.

The parser prompt includes compact, layout-independent demonstrations for
shared name/address evidence and preservation of payment states and dates.
Table row/column association is enforced locally rather than taught through a
sample invoice.

Grounding also performs conservative evidence-backed normalization. Common
decimal formats become canonical decimal strings; supported dates become ISO;
and currency codes are uppercased. `OVERDUE`, `PARTIALLY PAID`, and `REFUNDED`
remain unchanged source facts. Validation—not extraction—decides whether the
current Fakturama workflow supports them. When a claimed
postal/city span has a recognized country-compatible format such as
`10117 Berlin`, local code may split it into ZIP `10117` and city `Berlin`, with
both values retaining the same OCR evidence. Conflicting, unfamiliar, or
explicitly ambiguous address claims are never repaired automatically.

## Run artifacts

Every image run creates its directory before OCR:

```text
runs/<workflow-id>/
  evidence.json
  extraction.json
  validation.json
  checkpoint.json
  events.jsonl
  review.json                 # only when review is required
  review-assets/              # relevant image crops when available
  screenshots/                # workflow screenshots when supported
```

`evidence.json` is persisted immediately after OCR, before the remote call.
`extraction.json` contains both the nullable LLM draft and the grounded
`OrderInput` (or `null` when authorization failed). `validation.json` contains
stable issue codes, paths, severities, and messages. `checkpoint.json` records
the durable state.

## Developer diagnostics

These commands help develop and inspect the system; normal operation does not
require command choreography.

Extract without Fakturama writes:

```powershell
fakturama-automation extract ".\order.jpg"
```

Validate existing JSON:

```powershell
fakturama-automation validate ".\examples\order.json"
```

Run the business workflow against the in-memory gateway:

```powershell
fakturama-automation simulate ".\examples\order.json"
```

Inspect Fakturama UI Automation controls before real writes:

```powershell
fakturama-automation inspect-uia `
  --executable "C:\Program Files\Fakturama2\Fakturama.exe" `
  --profile ".\config\fakturama-2.2.0-en.json" `
  --output ".\runs\uia-tree.json"
```

The old supplier/coordinate parser remains available only for diagnosis:

```powershell
fakturama-automation extract ".\order.jpg" --parser rules
```

It always routes to review and cannot authorize real UI writes. `--parser auto`
is retained as a compatibility alias for the trusted text-LLM path; it no
longer invokes coordinate rules.

PP-StructureV3 remains optional because it is slow on CPU and is not required
for the trusted flow:

```powershell
fakturama-automation extract ".\order.jpg" --layout-analysis
```

## Financial and business safety

All financial values use `Decimal` and `ROUND_HALF_UP` cent rounding:

```text
line net = quantity * unit net * (1 - discount / 100)
line VAT = round(line net * VAT / 100, 2)
gross total = total net + total VAT
```

Source line totals and document totals must match locally calculated values
within one cent. High OCR confidence never overrides a contradiction.

Debtors, Products, VAT records, and payment methods use exact business rules.
Multiple exact matches require review; the workflow never chooses the first
ambiguous record. Products use exact SKU identity. The Invoice is created only
as a follow-up document from the saved Order.

## Testing

```powershell
python -m pytest -q
```

Unit tests do not require Paddle model downloads, a live LLM, or Fakturama.
Real-image OCR/LLM evaluation and real UIA testing are separate integration
activities.

## Known limitations

- PaddleOCR uses one configured recognition model per primary pass. Select the
  best-fit model through `OCR_LANGUAGES`; automatic script detection and
  critical-region fallback passes are future evaluation-driven work.
- Human review is durable JSON plus crops, not a review web application.
- Automatic cross-process resume is intentionally disabled. If an Order may
  already have been saved, do not blindly rerun record creation.
- Extraction preserves `PAID`, `UNPAID`, `OVERDUE`, `PARTIALLY PAID`, and
  `REFUNDED`. The current Fakturama writer authorizes only `PAID` and `UNPAID`;
  the other states route to review until an explicit UI business mapping exists.
- The checked-in Fakturama 2.2.0 English UIA profile must be validated against
  the installed locale, workspace, visible columns, and SWT accessibility tree.
- The Main address is assigned the Invoice role. When billing and delivery are
  identical, it is also assigned the Delivery role. When they differ, the
  current recording workflow continues without creating or assigning a second
  address; the extracted delivery address remains in the audit artifacts but is
  intentionally not written to the newly created Debtor.
- Fakturama startup readiness is configured in that profile. The default waits
  up to 300 seconds for a cold start, discovers the SWT top-level handle through
  Win32, reconnects it through UIA, and ignores splash screens until a known
  workspace action is accessible.
- PP-StructureV3 and EasyOCR are opt-in diagnostics, not default ensemble stages.
