# Roadmap

CrowdTensor's north star is open AI infrastructure that can use ordinary
machines for fault-tolerant inference and, later, broader AI workloads.

The project is deliberately staged. Each stage must leave behind runnable
commands, redacted evidence, and clear boundaries before the next layer is
claimed.

## Current Milestone

**Public Real-LLM Swarm Inference Beta**

The current beta proves a small real Hugging Face GPT model can be split across
two stage Miners behind a Coordinator-backed swarm route.

What is working today:

- Local `p2pd -> serve -> join stage0/stage1 -> generate` flow.
- Product-shaped `public-real-llm-swarm-beta` release gate.
- Stage-aware scheduling and distinct stage Miner validation.
- Read-only split inference with decoded-token baseline checks.
- KV cache evidence and 16-token generation readiness.
- Stage failure requeue proofs for controlled live-style runs.
- Redacted public evidence, support bundles, and cleanup paths.
- Optional CUDA tiny-model stage execution when explicitly enabled.

This is an engineering beta, not production Swarm Inference and not
Hivemind-level large-model serving.

## Near Term

**Make the beta easier to run on real machines.**

- Shorten the two-machine setup path.
- Improve join packs for non-expert Miner operators.
- Add first-class operator invite flows for role-scoped generation,
  accounting, and audit users while keeping plaintext tokens out of public
  reports.
- Extend role-scoped operator tokens into tenant/project-scoped policies and
  user-facing operator management.
- Extend private Miner join policy enforcement, claim-rate limits, accounting
  rows, and draft reward summaries into settlement exports and operator-visible
  trust state.
- Make logs, diagnosis codes, and support bundles easier to read.
- Keep printed next commands copyable while surfacing tokens and peer secrets
  as environment requirements rather than report contents.
- Keep private tokens and generated runtime state out of shareable artifacts by
  default.
- Add more operator-facing examples for local, LAN, VPN, and temporary public
  demos.
- Keep remote-Miner onboarding explicit about reachability: a Coordinator needs
  a Miner-facing URL via public HTTPS, a tunnel, VPN, reverse proxy, or trusted
  LAN before Miner hosts outside the local machine can join.

**Harden the public swarm route.**

- Keep the Coordinator as the current authority for sessions and validation.
- Improve discovery reliability and endpoint health checks.
- Continue failure-mode testing for killed or delayed stage Miners.
- Preserve deterministic evidence for every user-facing release gate.

**Improve the real-model path.**

- Keep `sshleifer/tiny-gpt2` as the safe default.
- Add clearer model compatibility checks before a run starts.
- Expand small-model variants only when correctness and artifact safety remain
  easy to verify.
- Keep CPU as the default path; keep CUDA opt-in and fail-closed.

## Mid Term

**Better networking.**

- Move from controlled discovery toward stronger P2P-lite behavior.
- Evaluate relay, NAT traversal, identity, and signed peer records.
- Keep production DHT/libp2p claims gated behind real external evidence.

**Better serving ergonomics.**

- Improve streaming generation output.
- Add better session inspection and cancellation commands.
- Make Miner capability selection and health easier to understand.
- Keep a clean separation between public user commands and maintainer release
  gates.

**Browser and edge experiments.**

- Continue WebRTC tensor transport experiments.
- Continue Web Worker and browser Miner probes.
- Evaluate WebGPU only after the native route remains stable.

## Later

These are intentionally later because they require stronger reliability,
security, and economics than the current beta provides:

- Permissionless public Miner admission.
- Large open-weight model sharding.
- Production GPU pooling.
- Training workloads across untrusted home machines.
- Incentives, staking, accounting, and reputation.
- Strong adversarial validation for unknown public Miners.

## Non-Goals For The Current Beta

CrowdTensor is not currently:

- A drop-in replacement for Hivemind, Petals, vLLM, or llama.cpp.
- A production inference API for arbitrary public prompts.
- A finished P2P network with DHT/NAT traversal.
- A large-model serving platform.
- A tokenized compute marketplace.

The current goal is narrower and more useful: prove that small real-model swarm
inference can be routed, recovered, validated, and explained in a way ordinary
contributors can run.
