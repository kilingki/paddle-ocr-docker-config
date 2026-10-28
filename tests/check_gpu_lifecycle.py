"""Independent real-runtime validation. No InferSwap registration or T5 mutation."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, http.client, json, os, shutil, subprocess
import threading, time
from runtime_support import ROOT, Ledger, json_request, load_fixtures, multipart, write
from check_api import check_api
from benchmark_concurrency import benchmark

BASE="http://127.0.0.1:8020"
OTHER={"qwen-vlm":8010,"qwen-asr":8080,"qwen-fa":8090}
CONTAINER="paddle-ocr-docker-config-paddle-ocr-runtime-1"


def command(args, timeout=60, env=None, check=True):
    result=subprocess.run(args,cwd=ROOT,env=env,text=True,capture_output=True,timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(str(args)+": "+(result.stdout+result.stderr)[-3000:])
    return result


def other_states():
    return {k:json_request(f"http://127.0.0.1:{p}","/control/status")[1] for k,p in OTHER.items()}


def gpu():
    return int(command(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"]).stdout.strip().splitlines()[0])


def status():
    code,value=json_request(BASE,"/control/status")
    if code!=200:raise RuntimeError("runtime status failed")
    return value


def wait_idle(timeout=45):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        value=status()
        if value["active_requests"]==0:
            return value
        time.sleep(.1)
    raise RuntimeError("runtime active did not converge")


def control(kind):
    code,value=json_request(BASE,"/control/"+kind,{},timeout=180)
    if code!=200:raise RuntimeError(f"{kind}: {code} {value}")
    return value


def configure(concurrency, request_timeout=120):
    current=status()
    if current["active_requests"] or current["residency"]!="not_resident":
        raise RuntimeError("configuration change requires confirmed unloaded runtime")
    env=dict(os.environ,MAX_ACTIVE_REQUESTS=str(concurrency),REQUEST_TIMEOUT=str(request_timeout))
    command(["docker","compose","up","-d","--no-build","--force-recreate","paddle-ocr-runtime"],env=env)
    end=time.monotonic()+20
    while time.monotonic()<end:
        try:
            current=status()
            if current["state"]=="unloaded" and current["active_requests"]==0:
                return
        except Exception:pass
        time.sleep(.25)
    raise RuntimeError("new controller did not start unloaded")


def private_metrics():
    code=("import urllib.request,json,re;"
          "s=urllib.request.urlopen('http://127.0.0.1:18000/metrics',timeout=10).read().decode();"
          "print(json.dumps({n:sum(map(float,re.findall(r'^vllm:'+n+r'(?:\\{[^\\n]*\\})?\\s+([0-9.eE+-]+)',s,re.M)))"
          " for n in ['num_requests_running','num_requests_waiting','kv_cache_usage_perc']}))")
    return json.loads(command(["docker","exec",CONTAINER,"python3","-c",code],timeout=15).stdout)


def cancel_pair(fixtures, ledger, out):
    cancelled=next(f for f in fixtures if f["task"]=="table")
    other=next(f for f in fixtures if f["task"]=="spotting")
    row=ledger.reserve(cancelled,"cancel_during_other_request")
    body,headers=multipart(cancelled)
    connection=http.client.HTTPConnection("127.0.0.1",8020,timeout=180)
    try:
        connection.request("POST","/ocr",body=body,headers=headers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            # Wait for actual engine activity before closing the outer HTTP connection.
            end=time.monotonic()+10
            while time.monotonic()<end:
                if status()["active_requests"]>=1:
                    metrics=private_metrics()
                    if metrics["num_requests_running"]>=1:break
                time.sleep(.05)
            else:raise RuntimeError("cancel request never reached engine")
            future=pool.submit(ledger.infer,BASE,other,"cancel_unrelated_survivor")
            end=time.monotonic()+10
            while time.monotonic()<end and status()["active_requests"]<2:
                time.sleep(.01)
            before=status()
            code,busy=json_request(BASE,"/control/unload",{})
            if code!=409 or busy["error"]["code"]!="BUSY":raise RuntimeError("active unload was not BUSY")
            closed=time.monotonic();connection.close()
            samples=[]
            while not future.done():
                samples.append(status());time.sleep(.05)
            survivor=future.result()
            terminal=wait_idle()
            if survivor["http_status"]!=200 or terminal["residency"]!="resident":raise RuntimeError("cancel affected another request/model")
        row.update(client_disconnected=True,engine_running_before_close=metrics,active_before=before,
                   seconds_to_idle=time.monotonic()-closed,active_samples=samples,terminal=terminal)
        write(out/"cancel_pair.json",row);write(ledger.out/"ledger.json",ledger.calls)
        return row
    finally:
        connection.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True)
    p.add_argument("--fixtures",default="tests/runtime_fixtures.json")
    args=p.parse_args()
    out=Path(args.out).resolve()
    if out.exists():raise RuntimeError("new validation must use a fresh run directory")
    out.mkdir(parents=True)
    fixtures=load_fixtures(args.fixtures);ledger=Ledger(out,budget=48)
    before=other_states();write(out/"runtime_before.json",before)
    diag=json_request("http://127.0.0.1:8095","/v1/models")[1];write(out/"inferswap_before.json",diag)
    if diag.get("queue_depth")!=0 or any(v["active_requests"] for v in before.values()):
        raise RuntimeError("existing active/queued work; no transition")
    if any(v["state"] not in ("ready","unloaded") for v in before.values()):
        raise RuntimeError("nonterminal runtime state")
    if any(v["residency"]!="not_resident" for k,v in before.items() if k!="qwen-vlm"):
        raise RuntimeError("ASR/FA resident; not changing them")
    sources={}
    for path in [*ROOT.glob("ocr-api/app/*.py"),*ROOT.glob("tests/*.py"),ROOT/"Dockerfile",ROOT/"docker-compose.yml",ROOT/"configs/model_manifest.json"]:
        relative=str(path.relative_to(ROOT));sources[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
        target=out/"source_snapshot"/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    write(out/"manifest.json",{"sources":sources,"sequence_budget":48,"wall_budget_seconds":1800,
        "engine_execution_budget_seconds":900,"profiles":[1,2,4],"benchmark_rounds":2,
        "automatic_retries":0,"model_revision":"c5630abae1d940eafe0697512a0325494b02ab42",
        "fixtures":[{k:v for k,v in f.items() if k!="file"} for f in fixtures]})
    write(out/"image.json",json.loads(command(["docker","image","inspect","paddle-ocr-runtime:local"]).stdout))
    summary={"state":"failed"};samples=[];stop=threading.Event();foreign=threading.Event();start=time.monotonic()
    thread=None
    def sample():
        n=0
        while not stop.is_set():
            row={"time":time.monotonic()}
            try:
                row["gpu_used_mib"]=gpu();row["ocr"]=status()
                if n%5==0:
                    row["others"]=other_states()
                    if any(v["residency"]!="not_resident" or v["state"]=="loading" or v["active_requests"] for v in row["others"].values()):
                        foreign.set()
                        command(["docker","compose","stop","-t","45","paddle-ocr-runtime"],check=False)
            except Exception as exc:row["error"]=repr(exc)
            samples.append(row);n+=1;stop.wait(.2)
    windows=[];cycles=[]
    try:
        if before["qwen-vlm"]["state"]=="ready":
            code,value=json_request("http://127.0.0.1:8010","/control/unload",{},timeout=45)
            write(out/"qwen_unload.json",value)
            if code!=200:raise RuntimeError("Qwen unload failed")
        if any(v["residency"]!="not_resident" for v in other_states().values()):raise RuntimeError("GPU transition not cleared")
        summary["gpu_baseline_mib"]=gpu()
        thread=threading.Thread(target=sample,daemon=True);thread.start()
        for concurrency in (1,2,4):
            if time.monotonic()-start>1800 or foreign.is_set():raise RuntimeError("wall budget or foreign activity")
            configure(concurrency)
            t=time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                one=pool.submit(control,"load");two=pool.submit(control,"load")
                a,b=one.result(),two.result()
            if a!=b or a["state"]!="ready":raise RuntimeError("duplicate load result mismatch")
            load_seconds=time.monotonic()-t
            before_prepare=status();command(["./prepare-inferswap"]);prep=status()
            if before_prepare!=prep:raise RuntimeError("prepare changed ready runtime")
            check_api(BASE,fixtures,ledger,f"warmup_c{concurrency}")
            windows.extend(benchmark(BASE,fixtures,ledger,concurrency,rounds=2))
            write(out/"concurrency.json",windows)
            if concurrency==4:cancel_pair(fixtures,ledger,out)
            wait_idle();t=time.monotonic();terminal=control("unload")
            cycles.append({"concurrency":concurrency,"load_seconds":load_seconds,
                           "unload_seconds":time.monotonic()-t,"terminal":terminal,"gpu_after_mib":gpu()})
            write(out/"cycles.json",cycles)
            (out/f"c{concurrency}_server.log").write_text(command(["docker","logs",CONTAINER],check=False).stdout+command(["docker","logs",CONTAINER],check=False).stderr)
            print(json.dumps({"concurrency":concurrency,"windows":[w["seconds"] for w in windows if w["concurrency"]==concurrency]}),flush=True)
        # Choose the fastest level whose outputs reproduce the C1 reference for every task/round.
        reference={r["body"]["task"]:r["body"]["native_text"] for r in ledger.calls if r["label"]=="c1_r0"}
        candidates=[]
        differences={}
        for c in (1,2,4):
            rows=[r for r in ledger.calls if r["label"].startswith(f"c{c}_r")]
            changed=[r["index"] for r in rows if r["body"]["native_text"]!=reference[r["task"]]]
            differences[str(c)]=changed
            if not changed:
                candidates.append(c)
        rates={c:sum(w["requests_per_second"] for w in windows if w["concurrency"]==c)/2 for c in (1,2,4)}
        chosen=max(candidates,key=lambda c:rates[c]) if candidates else 1
        summary.update(operating_concurrency=chosen,output_differences_from_c1=differences,
                       selection_rule="fastest repeatedly exact native outputs against C1; otherwise conservative1")
        # Two more load/infer/unload cycles at selected C, plus the selected profile's prior cycle.
        for repeat in range(2):
            configure(chosen);t=time.monotonic();control("load");load_time=time.monotonic()-t
            result=ledger.infer(BASE,fixtures[1],f"selected_lifecycle_{repeat}")
            if result["http_status"]!=200:raise RuntimeError("selected repeated inference failed")
            wait_idle();t=time.monotonic();control("unload")
            cycles.append({"selected":chosen,"repeat":repeat,"load_seconds":load_time,
                           "unload_seconds":time.monotonic()-t,"gpu_after_mib":gpu()})
            write(out/"cycles.json",cycles)
        # A real request deadline should abort, retain active until cleanup, and keep the model usable.
        configure(chosen,request_timeout=1)
        control("load")
        result=ledger.infer(BASE,next(f for f in fixtures if f["task"]=="table"),"deadline")
        write(out/"deadline.json",result)
        if result["http_status"]!=504:raise RuntimeError("deadline case did not time out as expected")
        wait_idle();control("unload")
        configure(chosen)
        summary["state"]="runtime_validation_passed"
    except Exception as exc:
        summary["error"]=repr(exc);print("VALIDATION ERROR",repr(exc),flush=True)
    finally:
        stop.set()
        if thread:thread.join(timeout=50)
        write(out/"samples.json",samples)
        try:
            wait_idle();control("unload")
        except Exception:
            command(["docker","compose","stop","-t","45","paddle-ocr-runtime"],check=False)
        summary["gpu_after_ocr_mib"]=gpu()
        summary["foreign_activity"]=foreign.is_set()
        summary["sequence_reservations"]=len(ledger.calls)
        summary["wall_seconds_before_restore"]=time.monotonic()-start
        write(out/"summary.json",summary)
        current=other_states()
        if (before["qwen-vlm"]["state"]=="ready" and current["qwen-vlm"]["state"]!="ready"
                and not any(v["residency"]!="not_resident" or v["active_requests"] for v in current.values())):
            code,value=json_request("http://127.0.0.1:8010","/control/load",{},timeout=600)
            write(out/"qwen_restore.json",value)
            if code!=200:raise RuntimeError("Qwen restore failed")
        write(out/"runtime_after.json",other_states());write(out/"ocr_after.json",status())
        write(out/"inferswap_after.json",json_request("http://127.0.0.1:8095","/v1/models")[1])
        summary["qwen_restored"]=other_states()["qwen-vlm"]["state"]==before["qwen-vlm"]["state"]
        summary["wall_seconds_total"]=time.monotonic()-start
        write(out/"summary.json",summary);print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
