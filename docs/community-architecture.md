# Community Training Architecture

The Community RC separates five boundaries.

```text
owner CLI/API
    |
Community workflow + role gateway
    |
durable Coordinator journal and scheduler
    |
Provider adapters (CPU, CUDA, optional JAX TPU)
    |
versioned Model Adapter -> stage runtime -> PEFT checkpoint/export
```

## Ownership

**Core runtime** owns atomic steps, stage activation/gradient handoff,
checkpoint content addressing, optimizer state, and independent adapter
reload. It never owns credentials or public release claims.

**Provider adapters** discover memory/performance capabilities and execute an
already-defined stage on CPU, CUDA, or JAX TPU. They do not decide model
semantics.

**Model Adapters** own model discovery, supported architecture checks, stage
partitioning/loading, LoRA target modules, resource estimates, checkpoint,
export, and reload. `model_adapter_v1.0` currently registers Qwen2 and SmolLM2.

**Control plane** owns admission, role policy, leases, fencing, replay
protection, scheduling, cancellation, quarantine, backpressure, audit events,
and cleanup.

**Evidence/release** consumes public-safe reports and cannot turn a blocker,
mock, queue event, or local smoke into live readiness.

The existing inference modules remain compatible and separate. The Community
training work does not rewrite their protocol or claim that training readiness
proves inference scalability.

## Atomic Step

1. Stage 0 produces an activation under a signed, expiring lease.
2. The next stage computes loss and the incoming gradient without stepping.
3. Stage 0 computes its parameter gradients without stepping.
4. Every stage receives the same commit generation and applies its optimizer.
5. The Coordinator records one ledger row only after every stage acknowledges.
6. Selected boundaries include adapter and optimizer state in a content-hashed
   checkpoint.

Coordinator restart invalidates leases but preserves the journal. Replacement
workers restore a committed checkpoint before receiving the next step.

## Evidence Language

`Kaggle logical multi-node` means distinct Kernel allocations and/or worker
processes coordinated as nodes. It never means independent physical hosts.
Physical multi-machine validation is outside this RC by explicit scope.
