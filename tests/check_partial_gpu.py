"""Final, two-sequence verification of partial-output error provenance."""
from pathlib import Path
import hashlib,json,os,shutil,time
from runtime_support import ROOT,Ledger,load_fixtures,json_request,write
from check_gpu_lifecycle import BASE,CONTAINER,command,status,control,configure,other_states,gpu,wait_idle
from check_gpu_recovery import assert_no_workers

def main():
    out=ROOT/"tests/outputs/20261028_runtime_validation_04"
    if out.exists():raise RuntimeError("fresh run required")
    out.mkdir(parents=True)
    ledger=Ledger(out,budget=3);before=other_states();write(out/"runtime_before.json",before)
    diag=json_request("http://127.0.0.1:8095","/v1/models")[1];write(out/"inferswap_before.json",diag)
    if diag.get("queue_depth") or any(v["active_requests"] for v in before.values()):raise RuntimeError("active work")
    if any(v["residency"]!="not_resident" for k,v in before.items() if k!="qwen-vlm"):raise RuntimeError("other residency")
    source={}
    for p in ROOT.glob("ocr-api/app/*.py"):
        rel=str(p.relative_to(ROOT));source[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
        dest=out/"source_snapshot"/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    write(out/"manifest.json",{"scope":"partial-output error preservation only; normal engine/lifecycle configuration unchanged",
        "sources":source,"sequence_budget":3,"previous_reserved":51,"total_budget":56,"request_timeout":15,"automatic_retries":0})
    write(out/"image.json",json.loads(command(["docker","image","inspect","paddle-ocr-runtime:local"]).stdout))
    summary={"state":"failed"};start=time.monotonic()
    try:
        if before["qwen-vlm"]["state"]=="ready":
            code,value=json_request("http://127.0.0.1:8010","/control/unload",{},timeout=45)
            write(out/"qwen_unload.json",value)
            if code!=200:raise RuntimeError("Qwen unload failed")
        summary["gpu_baseline_mib"]=gpu()
        configure(1,request_timeout=15)
        control("load")
        warm=ledger.infer(BASE,load_fixtures(ROOT/"tests/runtime_fixtures.json")[1],"final_warmup")
        if warm["http_status"]!=200:raise RuntimeError("warmup failed")
        file=ROOT/"tests/outputs/20261028_runtime_validation_03/max_table.png"
        fixture={"task":"table","file":file,"sha256":hashlib.sha256(file.read_bytes()).hexdigest()}
        result=ledger.infer(BASE,fixture,"partial_output_deadline")
        write(out/"partial_timeout.json",result)
        if result["http_status"]!=504:raise RuntimeError("expected deadline")
        error=result["body"]["error"]
        if not error.get("partial_result",{}).get("native_text"):raise RuntimeError("partial native output lost")
        if error["input"]["sha256"]!=fixture["sha256"] or error["task"]!="table":raise RuntimeError("error provenance mismatch")
        wait_idle();write(out/"after_cancel.json",status())
        control("unload");write(out/"workers_after.json",assert_no_workers())
        summary["state"]="partial_output_gpu_passed"
    except Exception as exc:
        summary["error"]=repr(exc);print("FINAL ERROR",repr(exc),flush=True)
    finally:
        p=command(["docker","logs",CONTAINER],check=False);(out/"server.log").write_text(p.stdout+p.stderr)
        try:wait_idle();control("unload")
        except Exception:command(["docker","compose","stop","-t","45","paddle-ocr-runtime"],check=False)
        summary["gpu_after_mib"]=gpu();summary["sequence_reservations"]=len(ledger.calls)
        write(out/"summary.json",summary)
        current=other_states()
        if before["qwen-vlm"]["state"]=="ready" and current["qwen-vlm"]["state"]!="ready":
            if any(v["residency"]!="not_resident" or v["active_requests"] for v in current.values()):raise RuntimeError("restore not clear")
            code,value=json_request("http://127.0.0.1:8010","/control/load",{},timeout=600)
            write(out/"qwen_restore.json",value)
            if code!=200:raise RuntimeError("Qwen restore failed")
        configure(1)
        write(out/"ocr_after.json",status());write(out/"runtime_after.json",other_states())
        write(out/"inferswap_after.json",json_request("http://127.0.0.1:8095","/v1/models")[1])
        summary["qwen_restored"]=other_states()["qwen-vlm"]["state"]==before["qwen-vlm"]["state"]
        summary["wall_seconds"]=time.monotonic()-start;write(out/"summary.json",summary);print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
