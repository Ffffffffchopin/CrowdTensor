#!/usr/bin/env python3
"""Create a role-scoped operator invite and hashed registry entry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.operator_invite import create_operator_invite  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a CrowdTensor role-scoped operator invite.")
    parser.add_argument("--registry", required=True, help="path to operator token registry JSON")
    parser.add_argument("--operator-id", required=True)
    parser.add_argument(
        "--role",
        action="append",
        default=[],
        help="operator role; repeat for multiple roles. Choices: owner, admin, accounting, auditor",
    )
    parser.add_argument("--label", default="")
    parser.add_argument("--token", default="", help="plaintext token to use; defaults to a generated random token")
    parser.add_argument("--replace", action="store_true", help="replace an existing registry entry for this operator_id")
    parser.add_argument(
        "--allowed-workload",
        action="append",
        default=[],
        help="allowed /admin/inference-sessions workload; repeat for multiple values",
    )
    parser.add_argument("--max-request-count", type=int, default=0)
    parser.add_argument("--max-decode-steps", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--max-active-sessions", type=int, default=0)
    parser.add_argument("--max-total-sessions", type=int, default=0)
    parser.add_argument("--rate-limit", type=int, default=0)
    parser.add_argument("--rate-window-seconds", type=float, default=0.0)
    parser.add_argument("--invite-file", default="", help="write the plaintext operator invite JSON to this private path")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roles = args.role or ["auditor"]
    try:
        invite = create_operator_invite(
            registry_path=Path(args.registry),
            operator_id=args.operator_id,
            roles=roles,
            label=args.label,
            token=args.token,
            replace=args.replace,
            allowed_workloads=args.allowed_workload,
            max_request_count=args.max_request_count,
            max_decode_steps=args.max_decode_steps,
            max_new_tokens=args.max_new_tokens,
            max_active_sessions=args.max_active_sessions,
            max_total_sessions=args.max_total_sessions,
            rate_limit=args.rate_limit,
            rate_window_seconds=args.rate_window_seconds,
            invite_file=Path(args.invite_file) if args.invite_file else None,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.json:
        print(json.dumps(invite, sort_keys=True))
        return

    print("CrowdTensor operator invite")
    print(f"  operator_id: {invite['operator_id']}")
    print(f"  roles: {', '.join(invite['roles'])}")
    print(f"  registry: {invite['registry']}")
    if invite["invite_file"]:
        print(f"  invite_file: {invite['invite_file']}")
    print(f"  token_hash: {invite['token_hash']}")
    print("  export:")
    print(f"    CROWDTENSOR_ADMIN_TOKEN={invite['env']['CROWDTENSOR_ADMIN_TOKEN']}")
    print("  coordinator:")
    print(f"    crowdtensor serve --operator-token-registry {invite['registry']} --run")
    print("  note: keep the invite/export output private; the registry stores only the token verifier.")


if __name__ == "__main__":
    main()
