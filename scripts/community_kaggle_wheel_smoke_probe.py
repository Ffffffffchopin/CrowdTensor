#!/usr/bin/env python3
"""Run a bounded single-CPU Kaggle clean-wheel preflight (not a full live gate)."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import shutil
import time
from pathlib import Path
from typing import Any

from crowdtensor.community_live_training import CommunityLiveCoordinator
from crowdtensor.community_security import scan_public_value
from crowdtensor.model_adapter import stable_hash
from scripts.community_kaggle_reliability_live_probe import (
    _authorized_kaggle_env,
    _hash,
    _start_live_tunnel,
    _start_server,
    _stop_server,
)
from scripts.community_kaggle_live_package import KAGGLE_RUNTIME_REQUIREMENTS
from scripts.training_cuda_kaggle_common import (
    authenticated_owner,
    delete_succeeded_or_absent,
    extract_kernel_ref,
    push_accepted,
    run_command,
    status_class,
)
from scripts.training_cuda_two_node_probe import ensure_cloudflared, stop_process
from scripts.training_heterogeneous_beta_live_probe import _free_port


SCHEMA = "crowdtensor_community_kaggle_wheel_smoke_v1"


def _source(coordinator: str, token: str) -> str:
    return f'''import hashlib,json,os,pathlib,re,subprocess,sys,urllib.request
coordinator={coordinator!r}
token={token!r}
working=pathlib.Path("/kaggle/working")
progress=working/"community_wheel_smoke_progress.json"
def phase(value): progress.write_text(json.dumps({{"phase":value,"public_artifact_safe":True}}))
phase("download")
request=urllib.request.Request(coordinator.rstrip("/")+"/v1/community-live/wheel",headers={{"x-crowdtensor-miner-token":token}})
with urllib.request.urlopen(request,timeout=180) as response:
 payload=response.read(); expected=response.headers.get("x-crowdtensor-wheel-sha256",""); name=response.headers.get("x-crowdtensor-wheel-filename","")
if "sha256:"+hashlib.sha256(payload).hexdigest()!=expected: raise RuntimeError("wheel_hash_invalid")
if not re.fullmatch(r"[A-Za-z0-9_.]+-[A-Za-z0-9_.!+]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+\\.whl",name): raise RuntimeError("wheel_name_invalid")
wheel=working/name; wheel.write_bytes(payload); phase("install")
runtime_root=pathlib.Path("/kaggle/temp") if pathlib.Path("/kaggle/temp").is_dir() else pathlib.Path("/tmp")
install_root=runtime_root/"ct-wheel-smoke-site"; install_root.mkdir(parents=True,exist_ok=True)
python=pathlib.Path(sys.executable)
runtime_requirements={list(KAGGLE_RUNTIME_REQUIREMENTS)!r}
subprocess.run([str(python),"-m","pip","install","--disable-pip-version-check","--target",str(install_root),"--no-deps",str(wheel)],check=True); phase("dependencies")
subprocess.run([str(python),"-m","pip","install","--disable-pip-version-check","--target",str(install_root),"--upgrade","--no-deps",*runtime_requirements],check=True)
env=dict(os.environ); env["PYTHONPATH"]=str(install_root)
subprocess.run([str(python),"-c","import torch,transformers,peft,safetensors,fastapi,accelerate; print('dependencies-ready')"],env=env,check=True)
workspace=working/"workflow"
cli=[str(python),"-m","crowdtensor.community_cli"]
commands=[cli+["init",str(workspace),"--json"],cli+["validate",str(workspace),"--json"],cli+["plan",str(workspace),"--json"]]
results=[subprocess.run(item,env=env,capture_output=True,text=True).returncode for item in commands]
check=subprocess.run([str(python),"-c","import crowdtensor,json,pathlib,os;print(json.dumps({{'version':crowdtensor.__version__,'under_install_root':str(pathlib.Path(crowdtensor.__file__).resolve()).startswith(str(pathlib.Path(os.environ['PYTHONPATH']).resolve()))}}))"],env=env,capture_output=True,text=True,check=True)
installed=json.loads(check.stdout); phase("completed")
report={{"schema":"crowdtensor_community_kaggle_wheel_smoke_kernel_v1","ok":all(item==0 for item in results) and installed.get("under_install_root") is True,"node_scope":"Kaggle logical single-node preflight","wheel_hash_verified":True,"valid_wheel_filename_verified":True,"wheel_installed_in_fresh_environment":True,"fresh_install_kind":"pip_target","fresh_install_root_per_kernel":True,"installed_version":installed.get("version"),"installed_package_under_install_root":installed.get("under_install_root") is True,"model_stack_import_verified":True,"runtime_requirements_exact_pins_verified":all("==" in item for item in runtime_requirements),"golden_command_count":3,"golden_commands_passed":all(item==0 for item in results),"workspace_import_used":False,"credential_values_public":False,"private_paths_public":False,"public_artifact_safe":True}}
(working/"community_wheel_smoke.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\\n")
print(json.dumps({{"ok":report["ok"]}}))
'''


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    private = output / ".private"
    private.mkdir(parents=True, exist_ok=True)
    wheel = Path(args.wheel).expanduser().resolve()
    wheel_hash = _hash(wheel.read_bytes())
    token = secrets.token_urlsafe(36)
    coordinator = CommunityLiveCoordinator(private / "state.json", run_id="wheel-smoke-" + secrets.token_hex(8), target_steps=1)
    server = thread = tunnel = None
    ref = ""
    cleanup = {"kernel_deleted": False, "server_stopped": False, "tunnel_stopped": False, "private_removed": False}
    report: dict[str, Any] = {}
    try:
        with _authorized_kaggle_env(args) as env:
            owner = authenticated_owner(env)
            if not owner:
                raise RuntimeError("wheel_smoke_auth_failed")
            port = _free_port()
            server, thread = _start_server(coordinator, port=port, token=token, wheel_path=wheel)
            cloudflared = ensure_cloudflared(private)
            tunnel, route = _start_live_tunnel(cloudflared, local_url=f"http://127.0.0.1:{port}", private_dir=private, token=token)
            slug = "ct-community-wheel-smoke-" + secrets.token_hex(5)
            package = private / "package"; package.mkdir()
            (package / "kernel.py").write_text(_source(route, token), encoding="utf-8")
            (package / "kernel-metadata.json").write_text(json.dumps({
                "id": f"{owner}/{slug}", "title": slug, "code_file": "kernel.py", "language": "python", "kernel_type": "script",
                "is_private": "true", "enable_gpu": "false", "enable_tpu": "false", "enable_internet": "true",
                "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [],
            }, indent=2), encoding="utf-8")
            push = run_command(["kaggle","kernels","push","-p",str(package)],env=env,timeout=300)
            ref = extract_kernel_ref(str(push.get("output_tail") or ""), f"{owner}/{slug}")
            if not push_accepted(push): raise RuntimeError("wheel_smoke_push_rejected")
            deadline=time.monotonic()+600
            state=""
            while time.monotonic()<deadline:
                status=run_command(["kaggle","kernels","status",ref],env=env,timeout=30)
                state=status_class(str(status.get("output_tail") or ""))
                if state in {"complete","failed"}: break
                time.sleep(5)
            destination=private/"output"; destination.mkdir(parents=True,exist_ok=True)
            result=run_command(["kaggle","kernels","output",ref,"-p",str(destination)],env=env,timeout=180)
            kernel_path=destination/"community_wheel_smoke.json"
            progress_path=destination/"community_wheel_smoke_progress.json"
            kernel=json.loads(kernel_path.read_text()) if kernel_path.is_file() else {}
            progress=json.loads(progress_path.read_text()) if progress_path.is_file() else {}
            report={
                "schema":SCHEMA,"ok":state=="complete" and kernel.get("ok") is True,"live_kernel_created":True,
                "full_live_gate":False,"gpu_used":False,"node_scope":"Kaggle logical single-node preflight",
                "kernel":kernel,"terminal_state":state,"last_phase":str(progress.get("phase") or ""),
                "wheel_sha256": wheel_hash,
                "credential_values_public":False,"coordinator_url_public":False,"private_paths_public":False,"public_artifact_safe":True,
            }
    except BaseException as exc:
        report={"schema":SCHEMA,"ok":False,"full_live_gate":False,"gpu_used":False,"wheel_sha256":wheel_hash,"blockers":[str(exc)[:160] if str(exc).startswith("wheel_smoke_") else "wheel_smoke_failed:"+type(exc).__name__],"credential_values_public":False,"private_paths_public":False,"public_artifact_safe":True}
    finally:
        if ref:
            try:
                with _authorized_kaggle_env(args) as env:
                    cleanup["kernel_deleted"]=delete_succeeded_or_absent(run_command(["kaggle","kernels","delete",ref,"-y"],env=env,timeout=120))
            except BaseException: pass
        else: cleanup["kernel_deleted"]=True
        cleanup["server_stopped"]=_stop_server(server,thread)
        cleanup["tunnel_stopped"]=stop_process(tunnel)
        shutil.rmtree(private,ignore_errors=True); cleanup["private_removed"]=not private.exists()
        report["cleanup"]=cleanup; report["cleanup_verified"]=all(cleanup.values())
        safety=scan_public_value(report); report["public_safety"]=safety; report["public_artifact_safe"]=safety["ok"] is True
        report["ok"]=bool(report.get("ok") and report["cleanup_verified"] and safety["ok"])
        report["content_hash"]=stable_hash(report)
        (output/"community_kaggle_wheel_smoke.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",required=True); parser.add_argument("--wheel",required=True)
    parser.add_argument("--kaggle-token-file",required=True); parser.add_argument("--kaggle-account-label",default=""); parser.add_argument("--kaggle-username",default="")
    parser.add_argument("--json",action="store_true"); args=parser.parse_args()
    report=run(args); print(json.dumps(report,sort_keys=True) if args.json else f"ok={report['ok']}")
    return 0 if report["ok"] else 1


if __name__=="__main__": raise SystemExit(main())
