#!/usr/bin/env python3
"""Run a bounded Kaggle GPU stage0 diagnostic without consuming a full gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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


SCHEMA = "crowdtensor_community_kaggle_gpu_stage0_diagnostic_v1"


def _source(coordinator: str, token: str, *, include_dual_stage: bool = False) -> str:
    return f'''import hashlib,json,math,os,pathlib,re,subprocess,sys,traceback,urllib.request
coordinator={coordinator!r}
token={token!r}
include_dual_stage={bool(include_dual_stage)!r}
working=pathlib.Path("/kaggle/working")
progress_path=working/"community_gpu_stage0_diagnostic_progress.json"
report_path=working/"community_gpu_stage0_diagnostic.json"
phase_value="started"
def phase(value):
 global phase_value
 phase_value=value
 progress_path.write_text(json.dumps({{"phase":value,"public_artifact_safe":True}}))
report={{"schema":"crowdtensor_community_kaggle_gpu_stage0_kernel_diagnostic_v1","ok":False,"diagnostic_only":True,"full_live_gate":False,"node_scope":"Kaggle logical single-node diagnostic","public_artifact_safe":True}}
try:
 phase("wheel_download")
 request=urllib.request.Request(coordinator.rstrip("/")+"/v1/community-live/wheel",headers={{"x-crowdtensor-miner-token":token}})
 with urllib.request.urlopen(request,timeout=180) as response:
  payload=response.read(); expected=response.headers.get("x-crowdtensor-wheel-sha256",""); name=response.headers.get("x-crowdtensor-wheel-filename","")
 if "sha256:"+hashlib.sha256(payload).hexdigest()!=expected: raise RuntimeError("wheel_hash_invalid")
 if not re.fullmatch(r"[A-Za-z0-9_.]+-[A-Za-z0-9_.!+]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+\\.whl",name): raise RuntimeError("wheel_name_invalid")
 wheel=working/name; wheel.write_bytes(payload)
 runtime_root=pathlib.Path("/kaggle/temp") if pathlib.Path("/kaggle/temp").is_dir() else pathlib.Path("/tmp")
 install_root=runtime_root/"ct-community-gpu-diagnostic-site"; install_root.mkdir(parents=True,exist_ok=True)
 python=pathlib.Path(sys.executable)
 runtime_requirements={list(KAGGLE_RUNTIME_REQUIREMENTS)!r}
 phase("wheel_install")
 subprocess.run([str(python),"-m","pip","install","--disable-pip-version-check","--target",str(install_root),"--no-deps",str(wheel)],check=True)
 subprocess.run([str(python),"-m","pip","install","--disable-pip-version-check","--target",str(install_root),"--upgrade","--no-deps",*runtime_requirements],check=True)
 sys.path.insert(0,str(install_root))
 phase("model_stack_import")
 import torch,transformers,peft,safetensors,fastapi,accelerate
 report["runtime_versions"]={{"torch":str(torch.__version__),"transformers":str(transformers.__version__),"peft":str(peft.__version__),"safetensors":str(safetensors.__version__),"accelerate":str(accelerate.__version__)}}
 report["runtime_requirements_exact_pins_verified"]=all("==" in item for item in runtime_requirements)
 report["cuda_device_count"]=int(torch.cuda.device_count())
 if not torch.cuda.is_available(): raise RuntimeError("cuda_unavailable")
 from crowdtensor.model_adapter import SmolLMModelAdapter
 from crowdtensor.smollm_training import _PassThroughLayer,_owned_lora_state,_tensor_state_hash
 adapter=SmolLMModelAdapter()
 phase("base_model_load")
 model=adapter.load_model(model_id=adapter.default_model_id,revision=adapter.default_revision,device="cuda:0",dtype=torch.float16,local_files_only=False,cache_dir=str(runtime_root/"ct-community-gpu-diagnostic-hf-cache"))
 phase("lora_apply")
 model=adapter.apply_lora(model,rank=8,alpha=16)
 phase("stage0_configure")
 causal=model.get_base_model(); decoder=causal.model; total_layers=len(decoder.layers); start,end=0,15
 for name,parameter in model.named_parameters():
  match=re.search(r"(?:^|\\.)layers\\.(\\d+)\\.",name)
  parameter.requires_grad=bool(match and start<=int(match.group(1))<end and "lora_" in name)
 for index in range(end,total_layers): decoder.layers[index]=_PassThroughLayer.create()
 decoder.config.num_hidden_layers=end; decoder.norm=torch.nn.Identity()
 trainable=[item for item in model.parameters() if item.requires_grad]
 if not trainable: raise RuntimeError("stage0_trainable_parameters_missing")
 optimizer=torch.optim.AdamW(trainable,lr=2e-4,weight_decay=0.0); model.train()
 initial=_tensor_state_hash(_owned_lora_state(model,start=start,end=end))
 phase("stage0_forward_backward")
 ids=torch.arange(8,device="cuda:0",dtype=torch.long).reshape(1,8)%49152
 mask=torch.ones_like(ids)
 optimizer.zero_grad(set_to_none=True)
 hidden=causal.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
 loss=hidden.float().square().mean()
 if not bool(torch.isfinite(loss).item()): raise RuntimeError("loss_non_finite")
 loss.backward()
 torch.nn.utils.clip_grad_norm_([item for item in model.parameters() if item.requires_grad],1.0)
 optimizer.step()
 final=_tensor_state_hash(_owned_lora_state(model,start=start,end=end))
 dual_stage={{}}
 if include_dual_stage:
  compatibility_disabled=bool(getattr(model,"_crowdtensor_outdated_optional_torchao_dispatch_disabled",False))
  del hidden,loss,optimizer,causal,model
  torch.cuda.empty_cache()
  phase("dual_gpu_two_stage_lora")
  dual_output=runtime_root/"ct-community-dual-stage-diagnostic"
  env=dict(os.environ); env["PYTHONPATH"]=str(install_root)
  completed=subprocess.run([str(python),"-m","crowdtensor.community_smollm_runner","--output-dir",str(dual_output),"--steps","2","--timeout-seconds","1200"],env=env,check=False,timeout=1250,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  dual_path=dual_output/"smollm_two_stage_lora_live.json"
  if completed.returncode!=0 or not dual_path.is_file():
   child=str(completed.stdout or "")[-4000:].replace(coordinator,"<private-url>").replace(token,"<redacted>")
   child=re.sub(r"https?://\\S+","<url>",child); child=re.sub(r"/(?:[^\\s/:]+/)+[^\\s:]+","<path>",child); child=re.sub(r"[A-Za-z0-9_-]{{32,}}","<opaque>",child)
   child="".join(char for char in child if char in "\\n\\t" or 32<=ord(char)<127)[-1600:]
   report["dual_stage_process_returncode"]=int(completed.returncode); report["dual_stage_output_hash"]="sha256:"+hashlib.sha256(str(completed.stdout or "").encode()).hexdigest(); report["dual_stage_output_summary_public"]=child
   raise RuntimeError("dual_gpu_two_stage_lora_failed")
  dual=json.loads(dual_path.read_text())
  dual_ok=bool(dual.get("ok") is True and dual.get("devices")==["cuda","cuda"] and int(dual.get("logical_stage_count") or 0)==2 and (dual.get("reload") or {{}}).get("adapter_reload_verified") is True)
  if not dual_ok: raise RuntimeError("dual_gpu_two_stage_lora_evidence_invalid")
  dual_stage={{"verified":True,"logical_stage_count":2,"devices":["cuda","cuda"],"committed_step_count":len(dual.get("committed_step_ids") or []),"both_stage_adapters_updated":dual.get("both_stage_adapters_updated") is True,"adapter_reload_verified":True,"report_hash":"sha256:"+hashlib.sha256(json.dumps(dual,sort_keys=True,separators=(",",":")).encode()).hexdigest()}}
 else:
  compatibility_disabled=bool(getattr(model,"_crowdtensor_outdated_optional_torchao_dispatch_disabled",False))
 phase("completed")
 report.update({{"ok":initial!=final and (not include_dual_stage or dual_stage.get("verified") is True),"wheel_hash_verified":True,"wheel_clean_install_verified":True,"fresh_install_kind":"pip_target","installed_under_temporary_root":True,"workspace_import_used":False,"model_stack_import_verified":True,"cuda_available":True,"cuda_device_count":int(torch.cuda.device_count()),"real_open_model_weights":True,"random_or_synthetic_weights_used":False,"stage0_model_loaded":True,"finite_loss_verified":True,"adapter_updated":initial!=final,"optimizer_step_applied":True,"outdated_optional_torchao_dispatch_disabled":compatibility_disabled,"dual_stage_requested":include_dual_stage,"dual_stage":dual_stage,"peak_cuda_memory_bytes":int(torch.cuda.max_memory_allocated()),"error_class":""}})
except BaseException as exc:
 text=str(exc).lower()
 category="cuda_oom" if "out of memory" in text else ("hf_or_network" if any(x in text for x in ("http","connection","huggingface","timeout")) else ("dependency" if isinstance(exc,(ImportError,ModuleNotFoundError)) else "runtime"))
 missing=str(getattr(exc,"name","") or "")
 missing=missing if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*",missing) else ""
 symbols=[]
 for pattern in (r"no module named ['\\\"]([A-Za-z_][A-Za-z0-9_.]*)",r"cannot import name ['\\\"]([A-Za-z_][A-Za-z0-9_]*)",r"from ['\\\"]([A-Za-z_][A-Za-z0-9_.]*)"):
  symbols.extend(re.findall(pattern,text))
 chain=[]; current=exc
 while current is not None and len(chain)<5:
  chain.append(type(current).__module__+"."+type(current).__name__); current=current.__cause__ or current.__context__
 summary=str(exc).replace(coordinator,"<private-url>").replace(token,"<redacted>")
 summary=re.sub(r"https?://\\S+","<url>",summary)
 summary=re.sub(r"/(?:[^\\s/:]+/)+[^\\s:]+","<path>",summary)
 summary=re.sub(r"(?i)(token|cookie|authorization|api[_-]?key)\\s*[=:]\\s*\\S+",r"\\1=<redacted>",summary)
 summary=re.sub(r"[A-Za-z0-9_-]{{32,}}","<opaque>",summary)
 summary="".join(char for char in summary if char in "\\n\\t" or 32<=ord(char)<127)[:400]
 frames=[{{"file":pathlib.Path(item.filename).name,"function":item.name,"line":int(item.lineno)}} for item in traceback.extract_tb(exc.__traceback__)[-8:]]
 report.update({{"ok":False,"failure_phase":phase_value,"error_class":"community_gpu_stage0_diagnostic_failed:"+type(exc).__name__,"error_category":category,"missing_module":missing,"import_symbols":sorted(set(symbols)),"exception_chain_types":chain,"error_summary_public":summary,"traceback_frames_public":frames,"error_message_hash":"sha256:"+hashlib.sha256(str(exc).encode()).hexdigest(),"credential_values_public":False,"private_paths_public":False}})
finally:
 report.update({{"credential_values_public":False,"coordinator_url_public":False,"raw_training_text_public":False,"token_ids_public":False,"activation_values_public":False,"gradient_values_public":False,"checkpoint_tensor_values_public":False,"private_paths_public":False,"public_artifact_safe":True}})
 report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\\n")
 print(json.dumps({{"ok":report["ok"],"phase":phase_value,"error_class":report.get("error_class","")}}))
'''


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    private = output / ".private"
    private.mkdir(parents=True, exist_ok=True)
    wheel = Path(args.wheel).expanduser().resolve()
    wheel_hash = "sha256:" + hashlib.sha256(wheel.read_bytes()).hexdigest()
    token = secrets.token_urlsafe(36)
    coordinator = CommunityLiveCoordinator(
        private / "state.json", run_id="gpu-diagnostic-" + secrets.token_hex(8), target_steps=1
    )
    server = thread = tunnel = None
    ref = ""
    cleanup = {
        "kernel_deleted": False,
        "server_stopped": False,
        "tunnel_stopped": False,
        "private_removed": False,
        "live_resources_left_running": True,
    }
    report: dict[str, Any] = {}
    try:
        with _authorized_kaggle_env(args) as env:
            owner = authenticated_owner(env)
            if not owner:
                raise RuntimeError("community_gpu_diagnostic_auth_failed")
            port = _free_port()
            server, thread = _start_server(coordinator, port=port, token=token, wheel_path=wheel)
            cloudflared = ensure_cloudflared(private)
            tunnel, route = _start_live_tunnel(
                cloudflared,
                local_url=f"http://127.0.0.1:{port}",
                private_dir=private,
                token=token,
            )
            slug = "ct-community-gpu-diagnostic-" + secrets.token_hex(5)
            package = private / "package"
            package.mkdir()
            (package / "kernel.py").write_text(
                _source(route, token, include_dual_stage=bool(args.include_dual_stage)),
                encoding="utf-8",
            )
            metadata = {
                "id": f"{owner}/{slug}",
                "title": slug,
                "code_file": "kernel.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": "true",
                "enable_tpu": "false",
                "enable_internet": "true",
                "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            }
            (package / "kernel-metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            push = run_command(["kaggle", "kernels", "push", "-p", str(package)], env=env, timeout=300)
            ref = extract_kernel_ref(str(push.get("output_tail") or ""), metadata["id"])
            if not push_accepted(push):
                raise RuntimeError("community_gpu_diagnostic_push_rejected")
            deadline = time.monotonic() + float(args.timeout_seconds)
            state = ""
            while time.monotonic() < deadline:
                status = run_command(["kaggle", "kernels", "status", ref], env=env, timeout=30)
                state = status_class(str(status.get("output_tail") or ""))
                if state in {"complete", "failed"}:
                    break
                time.sleep(5)
            destination = private / "output"
            kernel: dict[str, Any] = {}
            progress: dict[str, Any] = {}
            for attempt in range(1, 16):
                shutil.rmtree(destination, ignore_errors=True)
                destination.mkdir(parents=True, exist_ok=True)
                run_command(["kaggle", "kernels", "output", ref, "-p", str(destination)], env=env, timeout=180)
                kernel_path = destination / "community_gpu_stage0_diagnostic.json"
                progress_path = destination / "community_gpu_stage0_diagnostic_progress.json"
                if kernel_path.is_file():
                    kernel = json.loads(kernel_path.read_text(encoding="utf-8"))
                    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
                    break
                time.sleep(min(30, attempt * 2))
            report = {
                "schema": SCHEMA,
                "ok": state == "complete" and kernel.get("ok") is True,
                "diagnostic_only": True,
                "full_live_gate": False,
                "live_kernel_created": True,
                "gpu_used": True,
                "node_scope": "Kaggle logical single-node diagnostic",
                "terminal_state": state,
                "wheel_sha256": wheel_hash,
                "last_phase": str(progress.get("phase") or kernel.get("failure_phase") or ""),
                "kernel": kernel,
                "credential_values_public": False,
                "coordinator_url_public": False,
                "private_paths_public": False,
                "public_artifact_safe": True,
            }
    except BaseException as exc:
        blocker = str(exc)[:160] if str(exc).startswith("community_") else "community_gpu_diagnostic_failed:" + type(exc).__name__
        report = {
            "schema": SCHEMA,
            "ok": False,
            "diagnostic_only": True,
            "full_live_gate": False,
            "gpu_used": True,
            "wheel_sha256": wheel_hash,
            "blockers": [blocker],
            "credential_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
    finally:
        if ref:
            try:
                with _authorized_kaggle_env(args) as env:
                    cleanup["kernel_deleted"] = delete_succeeded_or_absent(
                        run_command(["kaggle", "kernels", "delete", ref, "-y"], env=env, timeout=120)
                    )
            except BaseException:
                pass
        else:
            cleanup["kernel_deleted"] = True
        cleanup["server_stopped"] = _stop_server(server, thread)
        cleanup["tunnel_stopped"] = stop_process(tunnel)
        shutil.rmtree(private, ignore_errors=True)
        cleanup["private_removed"] = not private.exists()
        cleanup["live_resources_left_running"] = not all(
            cleanup[key] for key in ("kernel_deleted", "server_stopped", "tunnel_stopped", "private_removed")
        )
        report["cleanup"] = cleanup
        report["cleanup_verified"] = not cleanup["live_resources_left_running"]
        safety = scan_public_value(report)
        report["public_safety"] = safety
        report["public_artifact_safe"] = safety["ok"] is True
        report["ok"] = bool(report.get("ok") and report["cleanup_verified"] and safety["ok"])
        report["content_hash"] = stable_hash(report)
        (output / "community_kaggle_gpu_stage0_diagnostic.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--kaggle-token-file", required=True)
    parser.add_argument("--kaggle-account-label", default="")
    parser.add_argument("--kaggle-username", default="")
    parser.add_argument("--timeout-seconds", type=float, default=1200)
    parser.add_argument("--include-dual-stage", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, sort_keys=True) if args.json else f"ok={report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
