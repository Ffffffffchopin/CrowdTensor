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

Build the maintainer readiness report before running longer acceptance:

```bash
crowdtensor release-ready --json
```

The command emits `release_readiness_v1` by wrapping `scripts/release_readiness_pack.py`. It combines Git metadata, the release gate, security preflight, and `demo_manifest_v1`, then blocks on diagnosis such as `git_dirty`, `release_gate_failed`, or `demo_manifest_failed`. Dirty worktrees block by default; `scripts/release_readiness_check.py --allow-dirty` is reserved for CI/development smoke validation and does not mean a dirty tree is tag-ready. The report is not production Swarm Inference readiness; it is an Alpha repository release gate for maintainers.

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

The release evidence JSON preserves safe runtime acceptance `summary_json` rows, aggregates `diagnosis_summary` / `diagnosis_by_check`, and carries remote `observability_summaries` such as `remote_compute_observability_v1` and `remote_demo_observability_v1` so reviewers can see stable operator triage and remote-demo observability beside the pass/fail state.

Generate a Support Bundle for troubleshooting the candidate:

```bash
python3 scripts/support_bundle.py \
  --runtime-report /tmp/crowdtensor_acceptance.json \
  --browser-report /tmp/crowdtensor_browser_acceptance.json \
  --release-evidence dist/release-evidence.json \
  --json-out /tmp/crowdtensor_support_bundle.json \
  --markdown-out /tmp/crowdtensor_support_bundle.md
```

The Support Bundle includes acceptance `diagnosis_summary` / `diagnosis_by_check` and safe remote `observability_summaries` when reports are provided, while still redacting token, lease, idempotency, weight, and delta-shaped fields.

Generate a local-loopback Demo Manifest when reviewers need one latest output artifact for this checkout:

```bash
python3 scripts/demo_manifest_pack.py \
  --output-dir dist/demo-manifest \
  --port 8914 \
  --request-count 4
```

The `demo_manifest_v1` JSON/Markdown indexes `runtime_matrix.json`, `remote_compute_evidence_v1`, `external_llm_evidence_v1`, `support_bundle`, and `remote_compute_observability_v1` summaries without expanding the project scope beyond the CPU-only local-loopback demo. The external LLM evidence section uses deterministic mock evidence by default and does not claim public prompt serving.

## Publish

Before tagging, update [CHANGELOG.md](../CHANGELOG.md) with the release version, verification status, known limitations, and user-facing changes.

Create the tag only after checks pass:

```bash
git tag -a v0.1.0a0 -m "CrowdTensorD v0.1.0a0"
git push origin main v0.1.0a0
```

Use `.github/release.yml` to categorize GitHub release notes. Attach `dist/release-evidence.json` and `dist/release-evidence.md` to the release when available.
