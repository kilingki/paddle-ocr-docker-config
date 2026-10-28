"""Targeted real-GPU checks after the recorded shutdown identity fix."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, http.client, json, os, shutil, subprocess, threading, time
from runtime_support import ROOT, Ledger, json_request, load_fixtures, write
from check_api import check_api
from check_gpu_lifecycle import BASE, CONTAINER, command, other_states, gpu, status, wait_idle, control, configure, cancel_pair


def assert_no_workers():
    code=("import psutil,json;"
          "print(json.dumps([p.info for p in psutil.process_iter(['pid','name','cmdline']) "
          "if p.info['name']=='vllm' or p.info['name'].startswith('VLLM')]))")
    value=json.loads(command(["docker","exec",CONTAINER,"python3","-c",code]).stdout)
    if value:raise RuntimeError("owned vLLM processes remain after unload")
    return value


def save_logs(out,label):
    p=command(["docker","logs",CONTAINER],check=False)
    (out/(label+".log")).write_text(p.stdout+p.stderr)


def make_stress_fixture(out):
    script=("from PIL import Image,ImageDraw,ImageFont; "
        "im=Image.new('RGB',(8192,2048),'white');d=ImageDraw.Draw(im);f=ImageFont.load_default(size=48);"
        "[(d.line((x*256,0,x*256,2047),fill='black',width=3)) for x in range(33)];"
        "[(d.line((0,y*64,8191,y*64),fill='black',width=3)) for y in range(33)];"
        "[(d.text((x*256+12,y*64+8),str(y*32+x).zfill(6),font=f,fill='black')) for y in range(32) for x in range(32)];"
        "im.save('/output/max_table.png')")
    command(["docker","run","--rm","--network","none","-v",str(out)+":/output","--entrypoint","python3","paddle-ocr-runtime:local","-c",script])
    file=out/"max_table.png"
    return {"task":"table","file":file,"path":str(file),"sha256":hashlib.sha256(file.read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);args=p.parse_args()
    out=Path(args.out).resolve()
    if out.exists():raise RuntimeError("use a fresh evidence run")
    out.mkdir(parents=True);ledger=Ledger(out,budget=16);fixtures=load_fixtures(ROOT/"tests/runtime_fixtures.json")
    before=other_states();write(out/"runtime_before.json",before)
    diag=json_request("http://127.0.0.1:8095","/v1/models")[1];write(out/"inferswap_before.json",diag)
    if diag.get("queue_depth")!=0 or any(s["active_requests"] for s in before.values()):raise RuntimeError("existing active work")
    if any(s["state"] not in ("ready","unloaded") for s in before.values()):raise RuntimeError("nonterminal existing runtime")
    if any(s["residency"]!="not_resident" for k,s in before.items() if k!="qwen-vlm"):raise RuntimeError("ASR/FA residency")
    sources={}
    for source in [*ROOT.glob("ocr-api/app/*.py"),*ROOT.glob("tests/*.py"),ROOT/"Dockerfile",ROOT/"docker-compose.yml",ROOT/"configs/model_manifest.json"]:
        relative=str(source.relative_to(ROOT));sources[relative]=hashlib.sha256(source.read_bytes()).hexdigest()
        target=out/"source_snapshot"/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    write(out/"manifest.json",{"reason":"targeted shutdown/failure recovery correction; successful C1/2/4 measurements preserved in runtime_validation_01",
        "sequence_budget":16,"total_runtime_stage_budget":56,"previous_reservations":38,"wall_budget_seconds":1800,
        "sources":sources,"automatic_retries":0,"operating_concurrency":1})
    write(out/"image.json",json.loads(command(["docker","image","inspect","paddle-ocr-runtime:local"]).stdout))
    stress=make_stress_fixture(out)
    stop=threading.Event();foreign=threading.Event();samples=[];thread=None;start=time.monotonic();summary={"state":"failed"}
    def sample():
        n=0
        while not stop.is_set():
            row={"time":time.monotonic()}
            try:
                row["gpu_used_mib"]=gpu()
                t=time.monotonic();row["ocr"]=status();row["status_latency_seconds"]=time.monotonic()-t
                if n%5==0:
                    row["others"]=other_states()
                    if any(s["state"]=="loading" or s["residency"]!="not_resident" or s["active_requests"] for s in row["others"].values()):
                        foreign.set();command(["docker","compose","stop","-t","45","paddle-ocr-runtime"],check=False)
            except Exception as exc:row["error"]=repr(exc)
            samples.append(row);n+=1;stop.wait(.2)
    cycles=[]
    def check_budget():
        if foreign.is_set() or time.monotonic()-start>1800:raise RuntimeError("wall budget/foreign activity")
    def unload_record(label,baseline,tload):
        wait_idle();t=time.monotonic();terminal=control("unload")
        record={"label":label,"load_seconds":tload,"unload_seconds":time.monotonic()-t,
                "baseline_mib":baseline,"after_mib":gpu(),"terminal":terminal,"workers_after":assert_no_workers()}
        cycles.append(record);write(out/"cycles.json",cycles);save_logs(out,label);print(json.dumps(record),flush=True)
    try:
        if before["qwen-vlm"]["state"]=="ready":
            code,value=json_request("http://127.0.0.1:8010","/control/unload",{},timeout=45)
            write(out/"qwen_unload.json",value)
            if code!=200:raise RuntimeError("Qwen unload failed")
        if any(s["residency"]!="not_resident" for s in other_states().values()):raise RuntimeError("GPU not clear")
        # Initial old container is stopped after the failed run; explicit recreate is authorized deployment.
        command(["docker","compose","up","-d","--no-build","--force-recreate","paddle-ocr-runtime"],
                env=dict(os.environ,MAX_ACTIVE_REQUESTS="1",REQUEST_TIMEOUT="120"))
        end=time.monotonic()+20
        while time.monotonic()<end:
            try:
                if status()["state"]=="unloaded":break
            except Exception:pass
            time.sleep(.2)
        else:raise RuntimeError("controller not prepared")
        summary["gpu_baseline_mib"]=gpu();thread=threading.Thread(target=sample,daemon=True);thread.start()
        configure(4);baseline=gpu();t=time.monotonic();control("load");tload=time.monotonic()-t
        # Warm lazy processor/kernel setup before measuring an in-flight cancellation.
        warm = ledger.infer(BASE,fixtures[1],"c4_initial_warmup")
        if warm["http_status"] != 200:
            raise RuntimeError("warmup failed")
        cancel_pair(fixtures,ledger,out)
        one=ledger.infer(BASE,fixtures[1],"length_limit",max_tokens=1)
        if one["http_status"]!=200 or one["body"]["finish_reason"]!="length" or one["body"]["technical_status"]!="incomplete":
            raise RuntimeError("length boundary incorrect")
        unload_record("c4_cancel_shutdown",baseline,tload)
        for repeat in range(3):
            check_budget();configure(1);baseline=gpu();t=time.monotonic();control("load");tload=time.monotonic()-t
            begin=time.monotonic()
            if repeat==0:
                check_api(BASE,fixtures,ledger,"final_image_four_tasks")
            else:
                result=ledger.infer(BASE,fixtures[1],f"selected_round_{repeat}")
                if result["http_status"]!=200:raise RuntimeError("repeat inference failed")
            if repeat==2:
                boundary=ledger.infer(BASE,stress,"maximum_image_table")
                write(out/"maximum_image.json",boundary)
                if boundary["http_status"]!=200:raise RuntimeError("maximum image request failed")
                if boundary["body"]["input"]["width"]!=8192 or boundary["body"]["input"]["height"]!=2048:
                    raise RuntimeError("boundary input dimensions changed")
            unload_record(f"selected_round_{repeat}",baseline,tload)
        check_budget();configure(1,request_timeout=1);baseline=gpu();t=time.monotonic();control("load");tload=time.monotonic()-t
        deadline=ledger.infer(BASE,next(f for f in fixtures if f["task"]=="table"),"request_deadline")
        write(out/"deadline.json",deadline)
        if deadline["http_status"]!=504:raise RuntimeError("deadline did not trigger")
        terminal=wait_idle()
        if terminal["state"]!="ready" or terminal["residency"]!="resident":raise RuntimeError("normal timeout unloaded model")
        # Graceful controller/container restart must terminate its worker and boot unloaded.
        save_logs(out,"deadline_before_restart")
        command(["docker","compose","restart","-t","45","paddle-ocr-runtime"])
        end=time.monotonic()+20
        while time.monotonic()<end:
            try:
                state=status()
                if state["state"]=="unloaded" and state["residency"]=="not_resident" and state["active_requests"]==0:break
            except Exception:pass
            time.sleep(.25)
        else:raise RuntimeError("restart did not leave unloaded controller")
        write(out/"restart.json",{"status":state,"workers":assert_no_workers(),"gpu_mib":gpu()})
        configure(1);summary["state"]="targeted_runtime_validation_passed";summary["operating_concurrency"]=1
    except Exception as exc:
        summary["error"]=repr(exc);save_logs(out,"failure");print("TARGETED ERROR",repr(exc),flush=True)
    finally:
        stop.set()
        if thread:thread.join(timeout=50)
        write(out/"samples.json",samples)
        try:wait_idle();control("unload")
        except Exception:command(["docker","compose","stop","-t","45","paddle-ocr-runtime"],check=False)
        summary["gpu_after_ocr_mib"]=gpu();summary["foreign_activity"]=foreign.is_set()
        summary["sequence_reservations"]=len(ledger.calls);summary["wall_before_restore"]=time.monotonic()-start
        write(out/"summary.json",summary)
        current=other_states()
        if before["qwen-vlm"]["state"]=="ready" and current["qwen-vlm"]["state"]!="ready":
            if any(s["residency"]!="not_resident" or s["active_requests"] for s in current.values()):
                raise RuntimeError("restore blocked by other active residency")
            code,value=json_request("http://127.0.0.1:8010","/control/load",{},timeout=600)
            write(out/"qwen_restore.json",value)
            if code!=200:raise RuntimeError("Qwen restore failed")
        # Start only the unloaded controller if failure required a container stop.
        try:final=status()
        except Exception:
            command(["docker","compose","up","-d","--no-build","--no-recreate","paddle-ocr-runtime"])
            time.sleep(2);final=status()
        write(out/"ocr_after.json",final);write(out/"runtime_after.json",other_states())
        write(out/"inferswap_after.json",json_request("http://127.0.0.1:8095","/v1/models")[1])
        summary["qwen_restored"]=other_states()["qwen-vlm"]["state"]==before["qwen-vlm"]["state"]
        summary["wall_seconds_total"]=time.monotonic()-start
        write(out/"summary.json",summary);print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
