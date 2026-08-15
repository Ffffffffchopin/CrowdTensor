# Provider Adapter Contract

A ProviderAdapter reports public-safe resources to a backend planner. It does
not acquire cloud accounts, start notebooks, define model layers, or bypass
training validation.

`ProviderSnapshot` records:

- provider/resource identifiers and a hashed machine identity;
- device type/count and bounded total/free memory;
- availability (`intermittent` or `stable_window`);
- supported dtypes, capabilities, and optional performance score;
- an optional stable-group ID for collective execution.

`crowdtensor/adapters/providers.py` maps retained CPU/CUDA/JAX-TPU capability
documents into generic snapshots. Discovery code lives in
`crowdtensor/adapters/capabilities.py`; importing the adapter does not import
Torch, JAX, Transformers, Accelerate, or DeepSpeed.

The planner rejects an incomplete stable group and never treats an
intermittent resource as a stable rank. Resource acquisition and credentials
remain provider-owned. Hosted notebook workers must be described as logical
nodes, not independently administered physical hosts.

New providers should implement the structural protocol in
`crowdtensor/core/plugins.py`, return deterministic public snapshots, and put
side-effectful acquisition behind an explicit user action and resource lock.
