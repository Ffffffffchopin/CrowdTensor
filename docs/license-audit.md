# License Audit

The repository declares Apache-2.0 in `LICENSE` and `pyproject.toml`. The
Community RC does not change that license.

The pinned SmolLM2-135M model declares Apache-2.0. Qwen2.5 model and dataset
licenses remain upstream assets and are not redistributed in the wheel,
sdist, container context, or release bundle. Users must review upstream terms
for their chosen model and dataset.

The release build emits a dependency inventory and CycloneDX SBOM from package
metadata. `UNKNOWN` dependency license entries are review findings, not an
automatic relicensing decision. No third-party model weights, Kaggle cookies,
tokens, or hosted-notebook files are included in the public bundle.

Optional container base images and MinIO are referenced by pinned release tags
and remain subject to their upstream licenses. The release manifest records
their identity hashes when built locally.
