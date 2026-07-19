# Community Example

Preview the complete workflow without allocating resources:

```bash
crowdtensor community init example-run --target-steps 100
crowdtensor community validate example-run --json
crowdtensor community plan example-run --json
crowdtensor community coordinator up example-run --dry-run --json
crowdtensor community miner join example-run --dry-run --json
crowdtensor community train example-run --dry-run --json
crowdtensor community pause example-run --dry-run --json
crowdtensor community resume example-run --dry-run --json
crowdtensor community rebalance example-run --dry-run --json
crowdtensor community export example-run --dry-run --json
crowdtensor community stop example-run --dry-run --json
crowdtensor community cleanup example-run --dry-run --json
```

Public reports are written under `example-run/artifacts`; private runtime state
stays under `example-run/.crowdtensor/private`.
