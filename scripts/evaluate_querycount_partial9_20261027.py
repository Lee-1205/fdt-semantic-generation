"""One-pass read-only evaluation of the preserved QueryCount partial-9 checkpoint."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, hashlib, json, math, sys, time
from contextlib import nullcontext
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"research"),str(ROOT/"scripts")]
from fdt_local.data import collate, make_split, oracle, graph_key
from fdt_local.model_masked import Config
from fdt_local.objectives import compute_metrics
from fdt_local.runner import configure_determinism, source_digest
from fdt_local.storage import CheckpointStore, ensure_space
from query_bound_refiner_v1 import QueryBoundRefiner
import querycount_alignment_v1 as qc_data
import role8_query_mix_v1 as mix_data
import run_e2_grounding_20261026 as common
import learning_io_v2 as io
from probe_e1_readonly import compute_metrics as frozen_metrics

SCHEMA="fdt_querycount_partial9_readonly_plan_v1"
SOURCE_SHA="5494ec7c3cf9d8466d799d05fd0b782a101969dedd645678ecc52154dba93675"
MODEL_SHA="ce6ba1b3adf1e0b7ba1d3b8d7b17915e6514429cd6a518057dae1211958614db"
ALGO_SHA="d346ee6410ca7627fd63e4dbb82d04057c715cd7765a0eb33e28c9917910034a"
MAX_CALLS=720; WALL=120; RESERVE=30; BATCH=4; SEED=17
FEATURES={"tokens","roles","fact_mask","queries","query_mask"}
PANEL_NAMES=("primary_querymix_dev64","uniform_dev64","old_role_dev128","old_normal16","old_large16","paired_new_ids16",
 "querycount_dev_q01","querycount_dev_q04","querycount_dev_q08","querycount_dev_q16",
 "transfer_q01","transfer_q04","transfer_q08","transfer_q16")
PHYSICAL_NAMES=tuple(x for x in PANEL_NAMES if x!="querycount_dev_q16")
CONTROLS=("uniform_dev64","old_role_dev128","old_normal16","old_large16")
RELATIVE=("primary_querymix_dev64","paired_new_ids16","querycount_dev_q01","querycount_dev_q04","querycount_dev_q08","querycount_dev_q16",
 "transfer_q01","transfer_q04","transfer_q08","transfer_q16")

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
 return h.hexdigest()

def state_sha(state):
 h=hashlib.sha256()
 for n,t in sorted(state.items()): h.update(n.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

ALGORITHM_FIELDS=("model","optimizer","scaler","trainable_names","phase","phase_step","updates","attempts","exposure",
 "query_exposure","ground_exposure","gates","history","sampler_rng","noise_rng","torch_rng","cuda_rng","python_rng",
 "numpy_rng","scheduler","query_bound_runtime","querycount_sampler_state")
def _digest_value(h,v):
 if isinstance(v,torch.Tensor): h.update(b"T"+str(v.dtype).encode()+str(tuple(v.shape)).encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
 elif isinstance(v,dict):
  h.update(b"D")
  for k in sorted(v,key=lambda k:(type(k).__name__,repr(k))): _digest_value(h,k); _digest_value(h,v[k])
 elif isinstance(v,list): h.update(b"L"); [_digest_value(h,x) for x in v]
 elif isinstance(v,tuple): h.update(b"Q"); [_digest_value(h,x) for x in v]
 else: h.update((type(v).__name__+":"+repr(v)+";").encode())
def algorithm_sha(payload):
 h=hashlib.sha256(); _digest_value(h,{k:payload[k] for k in ALGORITHM_FIELDS}); return h.hexdigest()

def row_identity(row):
 return {k:row.get(k) for k in ("graph_key","facts","queries","labels","hops","n")}

def validate_declaration_config(plan):
 policy=io.load_json(plan["dependencies"]["config"])
 expected={"seed":SEED,"query_counts":[1,4,8,16],"panels":14,"physical_panels":13,
  "batch_size":BATCH,"rollout_steps":4,"optimizer_allowed":False,
  "checkpoint_write_allowed":False,"e2_final_files_accessed":False}
 for key,value in expected.items():
  if policy.get(key)!=value: raise ValueError(f"policy config mismatch: {key}")
 return {"declaration_config_path":plan["dependencies"]["config"],
  "declaration_config_sha256":plan["dependency_sha256"]["config"],
  "declaration_seed":policy["seed"],"declaration_query_counts":policy["query_counts"],
  "declaration_batch_size":policy["batch_size"],"declaration_rollout_steps":policy["rollout_steps"],
  "execution_model_config_source":"partial9 checkpoint contract.config",
  "execution_model_config":None}

def paired_rename(rows):
 mapping=[6,7,2,3,4,5,0,1]
 out=[]
 for row in rows:
  clone=json.loads(json.dumps(row))
  clone["n"]=8
  clone["facts"]=[[mapping[s],mapping[o],kind,phrase] for s,o,kind,phrase in row["facts"]]
  clone["queries"]=[[mapping[s],mapping[o]] for s,o in row["queries"]]
  clone["graph_key"]=graph_key(8,clone["facts"])
  out.append(clone)
 return out

def load_panel_rows(plan):
 artifact=Path(plan["querycount_artifact"])
 transfer=Path(plan["transfer_artifact"])
 rows={
  "primary_querymix_dev64":io.load_json(artifact/"dev_q16.json"),
  "uniform_dev64":io.load_json(plan["uniform_dev_path"]),
  "old_role_dev128":io.load_json(plan["old_role_path"]),
  "old_normal16":make_split("test",16,seed=60001),
  "old_large16":make_split("test",16,seed=80001,n_range=(7,8)),
 }
 rows["paired_new_ids16"]=paired_rename(rows["old_normal16"])
 for q in (1,4,8,16): rows[f"querycount_dev_q{q:02d}"]=io.load_json(artifact/f"dev_q{q:02d}.json")
 for q in (1,4,8,16): rows[f"transfer_q{q:02d}"]=io.load_json(transfer/f"q{q:02d}.json")
 if len(rows)!=14: raise ValueError("expected fourteen named panels before alias removal")
 for i,(a,b) in enumerate(zip(rows["primary_querymix_dev64"],rows["querycount_dev_q16"])):
  if row_identity(a)!=row_identity(b): raise ValueError(f"Q16/primary alias mismatch at row {i}")
 del rows["querycount_dev_q16"]
 return rows

def expected_counts(rows):
 physical=sum(len(v) for v in rows.values())
 batches=sum((len(v)+BATCH-1)//BATCH for v in rows.values())
 return {"named_panels":14,"physical_panels":13,"physical_rows":physical,"batches":batches,
  "expected_actual_encode_calls":batches,"expected_actual_denoise_calls":batches*4,
  "expected_actual_encode_denoise_calls":batches*5,"max_actual_encode_denoise_calls":MAX_CALLS,
  "expected_readout_calls":batches*2,"query_score_positions":sum(sum(len(r["queries"]) for r in v) for v in rows.values()),
  "q16_dev_is_alias":True}

def validate_panels(rows,baseline,plan):
 if tuple(rows.keys())!=PANEL_NAMES[:6]+PANEL_NAMES[6:9]+PANEL_NAMES[10:]:
  # dict insertion order is fixed by load_panel_rows; Q16 alias is not a physical key.
  raise ValueError("panel ordering/physical panel contract mismatch")
 if sum(map(len,rows.values()))!=560: raise ValueError("expected 560 physical panel rows")
 if set(baseline)!=set(PANEL_NAMES): raise ValueError("baseline must bind all fourteen panel names")
 for name,panel in rows.items():
  for row in panel:
   q=len(row.get("queries",[]))
   if q<1 or q>16 or len(row.get("labels",[]))!=q or len(row.get("hops",[]))!=q: raise ValueError(f"row target/query length mismatch in {name}")
   original=oracle(int(row["n"]),row["facts"],row["queries"])
   independent=qc_data.v2.independent_oracle(int(row["n"]),row["facts"],row["queries"])
   if original!=independent or original!=(row["labels"],row["hops"]): raise ValueError(f"oracle/BFS labels/hops mismatch in {name}")
   features,_,_=collate([row],Config(),"cpu")
   if set(features)!=FEATURES or int(features["query_mask"].sum())!=q: raise ValueError(f"five-input/mask contract mismatch in {name}")
 for name in PANEL_NAMES:
  if baseline[name].get("noise_seeds") != [17]: raise ValueError(f"baseline noise seed mismatch: {name}")
  if name not in plan["baseline_raw_sha256"]: raise ValueError(f"baseline raw hash missing: {name}")
  if sha256(plan["baseline_raw_paths"][name])!=plan["baseline_raw_sha256"][name]: raise ValueError(f"baseline raw changed: {name}")
  raw=io.load_json(plan["baseline_raw_paths"][name])
  if any(raw.get(k)!=v for k,v in baseline[name].items()): raise ValueError(f"baseline raw/summary mismatch: {name}")
  if len(raw.get("raw",[]))!=len(rows.get(name,[])) and name!="querycount_dev_q16": raise ValueError(f"baseline raw graph count mismatch: {name}")
  if name!="querycount_dev_q16":
   for i,(row,saved) in enumerate(zip(rows[name],raw["raw"])):
    norm=lambda value:json.loads(json.dumps(value,ensure_ascii=False))
    if (row.get("graph_key")!=saved.get("graph_key") or norm(row["facts"])!=saved.get("facts")
        or norm(row["queries"])!=saved.get("queries") or list(map(int,row["labels"]))!=saved.get("gold")):
     raise ValueError(f"baseline raw input identity mismatch: {name} row {i}")
 if baseline["querycount_dev_q16"]!=baseline["primary_querymix_dev64"]:
  raise ValueError("baseline Q16 alias metrics are not byte-identical to primary panel")
 return expected_counts(rows)

def output_bytes(root):
 total=0
 for path in Path(root).rglob("*"):
  if path.is_file() and not path.is_symlink(): total+=path.stat().st_size
 return total

class ActualBudget:
 def __init__(self,deadline,cap,stop,output=None,max_output=64*1024**2): self.deadline=deadline; self.cap=cap; self.stop=Path(stop); self.output=Path(output) if output else None; self.max_output=max_output; self.encode=0; self.denoise=0; self.readout=0
 def charge(self,kind):
  if time.time()>=self.deadline: raise TimeoutError("readonly assessment deadline")
  if self.stop.exists(): raise InterruptedError("owned supervisor STOP")
  if self.output and output_bytes(self.output)>=self.max_output: raise RuntimeError("owned output exceeded 64MiB cap")
  actual=self.encode+self.denoise
  if actual>=self.cap: raise RuntimeError("actual encode/denoise cap exceeded")
  if kind=="readout" and self.readout>=280: raise RuntimeError("readout call cap exceeded")
  if kind=="encode": self.encode+=1
  elif kind=="denoise": self.denoise+=1
  elif kind=="readout": self.readout+=1
  else: raise ValueError(kind)

def score_panel(model,rows,cfg,budget,name,device="cuda"):
 ys=[]; masks=[]; hops=[]; ground_correct=ground_total=0; raw=[]; init_preds_all=[]; pred_all=[]
 for start in range(0,len(rows),BATCH):
  batch=rows[start:start+BATCH]
  features,gold,exposure=collate(batch,cfg,device)
  if set(features)!=FEATURES: raise ValueError("neural input feature whitelist mismatch")
  expected=[len(r["queries"]) for r in batch]
  if features["query_mask"].sum(-1).tolist()!=expected: raise ValueError("active query mask mismatch")
  budget.charge("encode")
  amp=torch.autocast("cuda",dtype=torch.float16) if str(device).startswith("cuda") else nullcontext()
  with torch.no_grad(),amp: encoded=model.encode(features)
  if not all(torch.isfinite(t).all().item() for t in encoded.values() if isinstance(t,torch.Tensor) and t.is_floating_point()): raise FloatingPointError("nonfinite encoding")
  budget.charge("readout"); init_logits=model.readout(encoded["init"]).float()
  ground_pred=encoded["ground"].argmax(-1); valid=features["fact_mask"].bool()
  ground_correct+=int((ground_pred[valid]==gold["ground"][valid]).sum()); ground_total+=int(valid.sum())
  generator=torch.Generator(device=device).manual_seed(SEED+start)
  noise=torch.randn((len(batch),16,cfg.z_dim),device=device,generator=generator,dtype=torch.float32)
  original_denoise=model.denoise
  def counted_denoise(z,t,enc,ablate=False): budget.charge("denoise"); return original_denoise(z,t,enc,ablate)
  model.denoise=counted_denoise
  try:
   amp=torch.autocast("cuda",dtype=torch.float16) if str(device).startswith("cuda") else nullcontext()
   with torch.no_grad(),amp: state=model.rollout(encoded,noise)
  finally: model.denoise=original_denoise
  if not torch.isfinite(state).all(): raise FloatingPointError("nonfinite rollout state")
  budget.charge("readout"); logits=model.readout(state).float()
  if not torch.isfinite(logits).all() or not torch.isfinite(init_logits).all(): raise FloatingPointError("nonfinite logits")
  ys.append(gold["labels"].cpu()); masks.append(features["query_mask"].cpu()); hops.append(gold["hops"].cpu())
  init_preds_all.append(init_logits.argmax(-1).cpu()); pred_all.append(logits.argmax(-1).cpu())
  for bi,row in enumerate(batch):
   q=len(row["queries"]); raw.append({"graph_index":start+bi,"graph_key":row.get("graph_key",row.get("canonical_graph_key")),
    "queries":row["queries"],"gold":list(map(int,row["labels"])),"initializer_logits":init_logits[bi,:q].cpu().tolist(),
    "diffusion_logits":logits[bi,:q].cpu().tolist(),"initializer_pred":init_logits[bi,:q].argmax(-1).cpu().tolist(),
    "diffusion_pred":logits[bi,:q].argmax(-1).cpu().tolist(),"ground_accuracy_row":float((ground_pred[bi,:len(row["facts"])]==gold["ground"][bi,:len(row["facts"])]).float().mean().item()),
    "noise_sha256":hashlib.sha256(noise[bi].cpu().contiguous().numpy().tobytes()).hexdigest()})
 init=torch.cat(init_preds_all); pred=torch.cat(pred_all); y=torch.cat(ys); mask=torch.cat(masks); hop=torch.cat(hops)
 return {"graphs":len(rows),"ground_accuracy":ground_correct/max(1,ground_total),
  "initializer":compute_metrics(init,y,mask,hop),"ddim_4":compute_metrics(pred,y,mask,hop),"raw":raw}

def _baseline_result(raw): return raw.get("results",{}).get("ddim_4",{})
def assessment(baseline,current):
 result={}; failures=[]
 for name in PANEL_NAMES:
  base=baseline[name]; now=current[name]
  b=_baseline_result(base); c=now["ddim_4"]
  accfloor=max(.94,b["accuracy"]-.02) if name in CONTROLS else b["accuracy"]-.02
  macrofloor=b["macro_f1"]-.05
  ok=c["accuracy"]>=accfloor and c["macro_f1"]>=macrofloor and now["ground_accuracy"]>=.95 and now["initializer"]["accuracy"]>=.95
  result[name]={"baseline_accuracy":b["accuracy"],"partial9_accuracy":c["accuracy"],"accuracy_delta":c["accuracy"]-b["accuracy"],
   "accuracy_floor":accfloor,"baseline_macro_f1":b["macro_f1"],"partial9_macro_f1":c["macro_f1"],"macro_floor":macrofloor,
   "relative_guard_pass":ok,"ground_accuracy":now["ground_accuracy"],"initializer_accuracy":now["initializer"]["accuracy"]}
  if not ok: failures.append(name)
 q1=_baseline_result(baseline["querycount_dev_q01"])["accuracy"]; q4=_baseline_result(baseline["querycount_dev_q04"])["accuracy"]
 c1=current["querycount_dev_q01"]["ddim_4"]["accuracy"]; c4=current["querycount_dev_q04"]["ddim_4"]["accuracy"]
 baseline_rate=((1-q1)+(1-q4))/2; partial_rate=((1-c1)+(1-c4))/2
 qdev={str(q):current[f"querycount_dev_q{q:02d}"]["ddim_4"]["accuracy"] for q in (1,4,8,16)}
 result["q1_q4_equal_weighted_error_rate"]={"baseline":baseline_rate,"partial9":partial_rate,
  "halved_if_baseline_nonzero":partial_rate<=baseline_rate/2 if baseline_rate>0 else None,
  "zero_error_ceiling":baseline_rate==0,"qdev_accuracy_by_count":qdev,
  "each_qdev_accuracy_at_least_99":all(v>=.99 for v in qdev.values())}
 return {"panels":result,"relative_guard_failures":failures,"one_assessment_only_no_two_consecutive_claim":True}

def preflight(plan_path,output,worker_path,strict=True):
 plan=io.load_json(plan_path)
 if plan.get("schema")!=SCHEMA or plan.get("authorized") not in (False,True): raise ValueError("assessment plan schema/auth marker")
 if plan.get("optimizer_allowed") is not False or plan.get("checkpoint_write_allowed") is not False or plan.get("e2_final_files_accessed") is not False: raise ValueError("readonly contract violated")
 if plan.get("output_dir")!=str(output.resolve()) or plan.get("worker_sha256")!=sha256(worker_path): raise ValueError("output/worker SHA mismatch")
 if (plan.get("wall_seconds"),plan.get("shutdown_reserve_seconds"),plan.get("max_actual_encode_denoise_calls"),plan.get("batch_size"),plan.get("noise_seeds"))!=(120,30,720,4,[17]): raise ValueError("fixed readonly budget mismatch")
 if source_digest()!=plan["runtime_source_sha256"] or plan["core_sha256"]!=plan["runtime_source_sha256"]: raise ValueError("core source digest mismatch")
 for name,path in plan["dependencies"].items():
  if sha256(path)!=plan["dependency_sha256"][name]: raise ValueError(f"dependency digest mismatch {name}")
 marker=Path(plan["checkpoint_marker"]); meta,payload=CheckpointStore(marker.parent).verify(marker)
 if meta["sha256"]!=SOURCE_SHA or plan["checkpoint_payload_sha256"]!=SOURCE_SHA: raise ValueError("partial9 checkpoint SHA mismatch")
 if (payload.get("updates"),payload.get("attempts"),payload.get("phase"))!=(9,9,2): raise ValueError("partial9 checkpoint counters mismatch")
 if algorithm_sha(payload)!=plan["algorithm_state_sha256"]: raise ValueError("partial9 algorithm-state digest mismatch")
 if state_sha(payload["model"])!=MODEL_SHA or plan["model_state_sha256"]!=MODEL_SHA: raise ValueError("partial9 model weights SHA mismatch")
 config=Config(**payload["contract"]["config"])
 policy_config=validate_declaration_config(plan)
 policy_config["execution_model_config"]={k:getattr(config,k) for k in config.__dataclass_fields__}
 configure_determinism()
 if strict:
  model=QueryBoundRefiner(config)
  if model.counts()["parameters"]!=102237197: raise ValueError("model parameter count mismatch")
  model.load_state_dict(payload["model"],strict=True)
  if state_sha(model.state_dict())!=MODEL_SHA: raise ValueError("strict partial9 weight import mismatch")
  del model
 baseline=io.load_json(plan["baseline_summary_path"])
 rows=load_panel_rows(plan); counts=validate_panels(rows,baseline,plan)
 if counts["physical_rows"]!=560 or counts["batches"]!=140 or counts["expected_actual_encode_denoise_calls"]!=700 or counts["query_score_positions"]!=5392: raise ValueError(f"physical panel counts mismatch: {counts}")
 if output.exists():
  allowed={"plan.json","readiness.json","code_snapshot","supervisor","supervisor.stdout.log","supervisor.stderr.log","worker.stdout.log","worker.stderr.log","worker_progress.json","diagnostic_result.json","incident.json","final_status.json"}
  unexpected={p.name for p in output.iterdir()}-allowed
  if unexpected: raise FileExistsError(f"unexpected output entries: {sorted(unexpected)}")
 return {"status":"CPU_PREFLIGHT_PASS","counts":counts,"baseline_names":list(PANEL_NAMES),"payload_sha256":meta["sha256"],
  "config_binding":policy_config}

def run(plan_path,output,worker_path):
 checked=preflight(plan_path,output,worker_path,strict=False); plan=io.load_json(plan_path)
 if plan.get("authorized") is not True: raise ValueError("inference requires separately reviewed/authorized plan")
 state=io.load_json(output/"supervisor/state.json"); started=float(state["invocation_started_unix"]); deadline=float(state["worker_deadline_unix"])
 budget=ActualBudget(deadline,MAX_CALLS,Path(plan["control_dir"])/"STOP",output,plan["max_output_bytes"])
 for reserved in ("worker_progress.json","diagnostic_result.json","incident.json","final_status.json"):
  if (output/reserved).exists(): raise FileExistsError(f"refusing overwrite {reserved}")
 meta,payload=CheckpointStore(Path(plan["checkpoint_marker"]).parent).verify(Path(plan["checkpoint_marker"]))
 configure_determinism(); cfg=Config(**payload["contract"]["config"])
 model=QueryBoundRefiner(cfg).to("cuda")
 model.load_state_dict(payload["model"],strict=True); model.eval()
 for p in model.parameters(): p.requires_grad_(False)
 before=state_sha(model.state_dict()); rng_cpu=torch.get_rng_state().clone(); rng_cuda=[x.clone() for x in torch.cuda.get_rng_state_all()]
 panels=load_panel_rows(plan); baseline=io.load_json(plan["baseline_summary_path"])
 current={}; progress_path=output/"worker_progress.json"
 def progress(stage): io.atomic_json(progress_path,{"worker_pid":os.getpid(),"stage":stage,"observed_unix":time.time(),
  "actual_encode_calls":budget.encode,"actual_denoise_calls":budget.denoise,"actual_model_calls":budget.encode+budget.denoise,
  "readout_calls":budget.readout,"updates":9,"attempts":9,"optimizer_used":False,"checkpoint_write_allowed":False})
 try:
  for name in PHYSICAL_NAMES:
   progress("evaluating:"+name); current[name]=score_panel(model,panels[name],cfg,budget,name)
  current["querycount_dev_q16"]=current["primary_querymix_dev64"]
  if budget.encode!=140 or budget.denoise!=560 or budget.encode+budget.denoise!=700 or budget.readout!=280: raise RuntimeError("actual call counters mismatch")
  after=state_sha(model.state_dict()); rng_cpu_after=torch.get_rng_state(); rng_cuda_after=torch.cuda.get_rng_state_all()
  if after!=MODEL_SHA or after!=before: raise RuntimeError("model or buffers changed")
  if not torch.equal(rng_cpu,rng_cpu_after) or any(not torch.equal(a,b) for a,b in zip(rng_cuda,rng_cuda_after)): raise RuntimeError("global Torch RNG changed")
  execution_config={k:getattr(cfg,k) for k in cfg.__dataclass_fields__}
  declaration_config=validate_declaration_config(plan)
  declaration_config["execution_model_config"]=execution_config
  result={"schema":"fdt_querycount_partial9_readonly_result_v1","status":"READONLY_PARTIAL9_ASSESSMENT_COMPLETE",
   "checkpoint_payload_sha256":meta["sha256"],"model_state_sha256_before_after": [before,after],
   "baseline_summary_sha256":sha256(plan["baseline_summary_path"]),"current_panels":current,
   "baseline_panels":baseline,"assessment":assessment(baseline,current),"physical_panel_names":list(PHYSICAL_NAMES),
   "named_panel_names":list(PANEL_NAMES),"config_binding":declaration_config,
   "alias":{"querycount_dev_q16":"primary_querymix_dev64","exact_cpu_identity_verified":True,"extra_forward_calls":0},
   "raw_graph_outputs":{k:v["raw"] for k,v in current.items() if k in PHYSICAL_NAMES},
   "budget":{"actual_encode_calls":budget.encode,"actual_denoise_calls":budget.denoise,
    "actual_encode_denoise_calls":budget.encode+budget.denoise,"cap":MAX_CALLS,"readout_calls":budget.readout,
    "physical_rows":560,"batches":140,"query_score_positions_with_prefix_overlap":5392,
    "optimizer_calls":0,"checkpoint_writes":0},
   "rng_unchanged":True,"all_results_preserved_once":True,"two_consecutive_guard_claimed":False,
   "e2_final_files_accessed":False,"optimizer_used":False,"checkpoint_written":False,
   "elapsed_seconds":time.time()-started,"interpretation":"Single descriptive assessment of partial9 weights; not the completed 128-step experiment, not a new blind benchmark, and not an automatic promotion."}
  dest=output/"diagnostic_result.json"; io.atomic_json(dest,result)
  if output_bytes(output)>plan["max_output_bytes"]: raise RuntimeError("total owned output exceeds 64MiB cap")
  progress("terminal"); return result
 except BaseException as exc:
  progress("terminal_error"); io.atomic_json(output/"incident.json",{"error":f"{type(exc).__name__}: {exc}","optimizer_used":False,"checkpoint_written":False}); raise

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--plan",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--preflight",action="store_true")
 a=ap.parse_args(); plan=a.plan.resolve(); output=a.output.resolve(); worker=Path(__file__).resolve()
 if a.preflight: preflight(plan,output,worker,strict=True); print("CPU_PREFLIGHT_PASS"); return 0
 run(plan,output,worker); return 0
if __name__=="__main__": raise SystemExit(main())
