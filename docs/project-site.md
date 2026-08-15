# Optional Project And Contributor UI

The packaged Volunteer Session can serve:

- `/` for the project page;
- `/join` for contributor enrollment;
- `/v1/volunteer/public-snapshot` for aggregate Campaign status;
- `/v1/volunteer/dashboard` for operational metrics;
- authenticated `/v1/volunteer/*` work routes.

Assets live in `crowdtensor/project_site` and
`crowdtensor/volunteer_dashboard`, so the wheel contains the UI. The UI is
optional and does not imply a CrowdTensor-hosted global Coordinator.

Create a user-owned Campaign and serve it on loopback:

```bash
crowdtensor volunteer campaign create-local ./campaign --target-rounds 2
python -m build
crowdtensor release prepare ./campaign-release
crowdtensor volunteer serve ./campaign \
  --host 127.0.0.1 --port 8789 \
  --public-url http://127.0.0.1:8789 \
  --release-dir ./campaign-release
```

For remote access, put a maintained HTTPS reverse proxy in front of the
loopback service. Enable trusted forwarded headers only with a configured
private proxy identity. External direct HTTP is rejected.

Public snapshots must not contain pairing codes, credentials, Cell identity,
raw training rows, token IDs, tensor values, or private paths. Browser tasks
are scheduler-calibration work unless a Campaign explicitly implements and
validates real browser model training; they must not be counted as model
updates.

Create pairing codes locally and transmit them through a private channel:

```bash
crowdtensor volunteer pair-code ./campaign --mode agent
crowdtensor volunteer pair-code ./campaign --mode browser
```

Codes are single-use. Campaign directories, invites, caches, and checkpoints
belong outside the repository.

The Native Agent is the primary contribution path. Its same-origin installer
verifies `SHA256SUMS`, checks the Campaign's resource estimate without
redeeming the code, and then runs bounded CPU/CUDA PEFT work. Browser work is
calibration only and never increments model-update or Adapter lineage counters.
