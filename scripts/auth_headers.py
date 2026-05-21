"""Small shared helpers for smoke-test authentication headers."""

from __future__ import annotations

import argparse
import os


MINER_TOKEN_ENV = "CROWDTENSOR_MINER_TOKEN"
MINER_TOKEN_HEADER = "x-crowdtensor-miner-token"
OBSERVER_TOKEN_ENV = "CROWDTENSOR_OBSERVER_TOKEN"
OBSERVER_TOKEN_HEADER = "x-crowdtensor-observer-token"


def resolve_miner_token(value: str | None = None) -> str:
    if value:
        return value
    return os.environ.get(MINER_TOKEN_ENV, "")


def resolve_observer_token(value: str | None = None) -> str:
    if value:
        return value
    return os.environ.get(OBSERVER_TOKEN_ENV, "")


def activate_miner_token(args: argparse.Namespace) -> str:
    token = resolve_miner_token(getattr(args, "miner_token", ""))
    if token:
        os.environ[MINER_TOKEN_ENV] = token
    return token


def activate_observer_token(args: argparse.Namespace) -> str:
    token = resolve_observer_token(getattr(args, "observer_token", ""))
    if token:
        os.environ[OBSERVER_TOKEN_ENV] = token
    return token


def add_miner_token_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--miner-token",
        default=os.environ.get(MINER_TOKEN_ENV, ""),
        help=f"shared Miner token; falls back to {MINER_TOKEN_ENV}",
    )


def add_observer_token_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--observer-token",
        default=os.environ.get(OBSERVER_TOKEN_ENV, ""),
        help=f"shared observer token for /state and /metrics; falls back to {OBSERVER_TOKEN_ENV}",
    )


def json_headers() -> dict[str, str]:
    headers = {"content-type": "application/json"}
    token = resolve_miner_token()
    if token:
        headers[MINER_TOKEN_HEADER] = token
    return headers


def observer_headers(*, json_content: bool = True) -> dict[str, str]:
    headers: dict[str, str] = {"content-type": "application/json"} if json_content else {}
    token = resolve_observer_token()
    if token:
        headers[OBSERVER_TOKEN_HEADER] = token
    return headers


def coordinator_env() -> dict[str, str]:
    env = dict(os.environ)
    miner_token = resolve_miner_token()
    observer_token = resolve_observer_token()
    if miner_token:
        env[MINER_TOKEN_ENV] = miner_token
    if observer_token:
        env[OBSERVER_TOKEN_ENV] = observer_token
    return env
