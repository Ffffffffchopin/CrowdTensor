# Release Process

This document defines the Alpha release candidate checklist. It is for maintainers preparing a public GitHub release or tag.

## Preflight

Start from a clean worktree on `main`:

```bash
git status --short --branch
```

Confirm package metadata and public docs:

```bash
python3 scripts/release_gate.py --json
python3 scripts/doctor.py --json
python3 scripts/security_preflight.py --json
```

## Verification

Run compile and unit checks:

```bash
python3 -m py_compile coordinator.py miner_cli.py scripts/*.py crowdtensor/*.py
python3 -m unittest discover -s tests -v
```

Run the non-browser runtime acceptance pack from a normal Linux shell with localhost networking:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8910 \
  --report /tmp/crowdtensor_acceptance.json
```

Run browser acceptance when Playwright and Chromium are available:

```bash
python3 scripts/browser_acceptance_pack.py \
  --allow-skip \
  --base-port 9310 \
  --report /tmp/crowdtensor_browser_acceptance.json
```

## Evidence

Generate release evidence after acceptance reports exist:

```bash
python3 scripts/release_evidence_pack.py \
  --runtime-report /tmp/crowdtensor_acceptance.json \
  --browser-report /tmp/crowdtensor_browser_acceptance.json \
  --json-out dist/release-evidence.json \
  --markdown-out dist/release-evidence.md
```

The release evidence JSON preserves safe runtime acceptance `summary_json` rows and aggregates `diagnosis_summary` / `diagnosis_by_check` so reviewers can see stable operator triage codes beside the pass/fail state.

Generate a Support Bundle for troubleshooting the candidate:

```bash
python3 scripts/support_bundle.py \
  --runtime-report /tmp/crowdtensor_acceptance.json \
  --browser-report /tmp/crowdtensor_browser_acceptance.json \
  --release-evidence dist/release-evidence.json \
  --json-out /tmp/crowdtensor_support_bundle.json \
  --markdown-out /tmp/crowdtensor_support_bundle.md
```

The Support Bundle includes acceptance `diagnosis_summary` / `diagnosis_by_check` when reports are provided, while still redacting token, lease, idempotency, weight, and delta-shaped fields.

## Publish

Before tagging, update [CHANGELOG.md](../CHANGELOG.md) with the release version, verification status, known limitations, and user-facing changes.

Create the tag only after checks pass:

```bash
git tag -a v0.1.0a0 -m "CrowdTensorD v0.1.0a0"
git push origin main v0.1.0a0
```

Use `.github/release.yml` to categorize GitHub release notes. Attach `dist/release-evidence.json` and `dist/release-evidence.md` to the release when available.
