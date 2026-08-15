"""Small training-first public command surface.

The numerical and volunteer subcommands own their parsers in their respective
modules.  Keeping this entry point lazy means a plain ``--help`` or workspace
operation does not import optional model, CUDA, TPU, or browser frameworks.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .version import __version__


_VOLUNTEER_ACTIONS = frozenset(
    {
        "cleanup",
        "campaign-status",
        "pair-code",
        "campaign",
        "serve",
        "operator",
        "contract",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crowdtensor",
        description="CrowdTensor collaborative training with resumable work units.",
    )
    actions = parser.add_subparsers(dest="command")
    actions.add_parser("train", help="Create, plan, run, or inspect a training workspace.")
    actions.add_parser("volunteer", help="Join or operate a volunteer training campaign.")
    actions.add_parser("adapters", help="List installed model adapter plugins.")
    actions.add_parser("release", help="Prepare or verify Campaign release artifacts.")
    actions.add_parser("version", help="Print the installed package version.")
    return parser


def _run_adapters(argv: Sequence[str]) -> None:
    """Retain the plugin discovery surface without importing old workflow code."""

    from .model_adapter import (
        adapter_registry_report,
        check_model_adapter_conformance,
        get_model_adapter,
    )

    parser = argparse.ArgumentParser(prog="crowdtensor adapters")
    actions = parser.add_subparsers(dest="action", required=True)
    listed = actions.add_parser("list")
    listed.add_argument("--json", action="store_true")
    checked = actions.add_parser("check")
    checked.add_argument("adapter_id")
    checked.add_argument("--json", action="store_true")
    args = parser.parse_args(list(argv))
    if args.action == "list":
        value = adapter_registry_report()
        value["ok"] = True
    else:
        value = check_model_adapter_conformance(get_model_adapter(args.adapter_id))
        value["ok"] = not bool(value.get("errors"))
    import json

    print(json.dumps(value, indent=2, sort_keys=True) if args.json else value)
    raise SystemExit(0 if value.get("ok", True) else 1)


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in {"-h", "--help"}:
        _parser().print_help()
        raise SystemExit(0)
    command = raw[0]
    if command == "version":
        print(__version__)
        raise SystemExit(0)
    if command == "train":
        from .core.cli import main as training_main

        # Campaign lifecycle and the managed contributor service remain in the
        # Volunteer CLI, but are reachable from the single training namespace.
        if len(raw) > 1 and raw[1] in _VOLUNTEER_ACTIONS:
            from .volunteer_training_cli import main as volunteer_main

            volunteer_main(raw[1:])
            return
        training_main(raw[1:])
        return
    if command == "volunteer":
        from .volunteer_training_cli import main as volunteer_main

        volunteer_main(raw[1:])
        return
    if command == "adapters":
        _run_adapters(raw[1:])
        return
    if command == "release":
        from .release import main as release_main

        release_main(raw[1:])
        return
    parser = _parser()
    parser.error(f"unknown command: {command}")


if __name__ == "__main__":
    main()
