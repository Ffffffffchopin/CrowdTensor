# Community RC Release Checklist

1. Run the focused Community tests and existing heterogeneous production
   regression suite on Python 3.11 and 3.12 where available.
2. Build wheel and sdist in isolation; install the wheel into a fresh venv and
   run `community init/validate/plan` without a workspace import.
   Kaggle logical nodes additionally install the exact direct pins in
   `requirements/community-kaggle-runtime.lock` under an isolated temporary
   `pip --target` root. Provider-managed Torch/CUDA wheels are not replaced.
3. Validate Dockerfile and Compose; build the local image and record its ID
   hash without publishing it.
4. Run the bounded chaos suite and real local MinIO API integration.
5. Preserve the original two failed full-gate records. The one-time 2026-07-17
   authorization amended the maximum from 2 to 3 and permitted only attempt 3.
   Attempt 3 completed 100 contiguous steps on one Kaggle CPU Kernel plus one
   T4x2 GPU Kernel in 578.93 seconds and then cleaned every resource. Its report
   is labeled `Kaggle logical multi-node`; no further gate is authorized.
6. Strict-check the 100-step CPU+GPU reliability report and dual-GPU SmolLM
   report. Confirm PEFT export and independent reload.
7. Generate SBOM, dependency/license inventory, artifact SHA-256 hashes, and
   the offline release manifest.
8. Run privacy-negative scanning over every public JSON/Markdown and the
   release bundle file list.
9. Run `community_maturity_rc_check.py --require-ready`.
10. Confirm no temporary Kaggle kernels, tunnel, local Coordinator, MinIO
    container, private package, or build container remains running.

This process creates an offline RC only. Publishing to PyPI, GitHub Releases,
or a container registry is a separate human decision and is not automated by
the Community goal.

Optional artifact signing uses an operator-controlled Sigstore or GPG key after
hash verification. Private signing keys must never enter this repository or a
Kaggle Kernel.

Kaggle images that expose an optional `torchao<0.16` are supported for ordinary
dense SmolLM/Qwen LoRA weights by disabling PEFT's incompatible optional
TorchAO dispatcher. The adapter fails closed instead of applying this shim when
the loaded weight type is actually supplied by TorchAO.
