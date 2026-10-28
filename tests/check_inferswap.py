"""Live InferSwap integration: all inference/control transitions originate at :8095.
The verifier never POSTs /control/load or /control/unload.
"""
from pathlib import Path
import argparse,hashlib,http.client,json,subprocess,threading,time,urllib.request,urllib.error
from runtime_support import ROOT,Ledger,load_fixtures,multipart,write,json_request

INFER="http://127.0.0.1:8095"
OCR="http://127.0.0.1:8020"
QWEN="http://127.0.0.1:8010"
CONTAINER="paddle-ocr-docker-config-paddle-ocr-runtime-1"


def command(args):
    p=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=60)
    if p.returncode:raise RuntimeError(str(args)+": "+p.stdout+p.stderr)
    return p.stdout


def get(base,path):
    code,value=json_request(base,path,timeout=5)
    if code!=200:raise RuntimeError(f"GET {base+path}: {code}")
    return value


def post_infer(model,body,headers,timeout=720):
    url=INFER+"/infer?model="+model
    request=urllib.request.Request(url,data=body,headers=headers)
    start=time.monotonic()
    try:
        with urllib.request.urlopen(request,timeout=timeout) as response:
            return {"http_status":response.status,"body":json.loads(response.read()),"seconds":time.monotonic()-start,"url":url}
    except urllib.error.HTTPError as exc:
        return {"http_status":exc.code,"body":json.loads(exc.read()),"seconds":time.monotonic()-start,"url":url}


def worker_processes():
    code="import psutil,json;print(json.dumps([p.info for p in psutil.process_iter(['pid','name']) if p.info['name']=='vllm' or p.info['name'].startswith('VLLM')]))"
    return json.loads(command(["docker","exec",CONTAINER,"python3","-c",code]))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",required=True);args=parser.parse_args()
    out=Path(args.out).resolve()
    if (out/"integration_manifest.json").exists():raise RuntimeError("existing integration run preserved")
    out.mkdir(parents=True,exist_ok=True)
    fixtures=load_fixtures(ROOT/"tests/runtime_fixtures.json");ledger=Ledger(out/"proxy",budget=8)
    initial={"ocr":get(OCR,"/control/status"),"qwen":get(QWEN,"/control/status"),"inferswap":get(INFER,"/v1/models")}
    write(out/"integration_before.json",initial)
    if (initial["ocr"]["state"]!="unloaded" or initial["ocr"]["active_requests"]
            or initial["qwen"]["active_requests"] or initial["inferswap"]["queue_depth"]):
        raise RuntimeError("not an idle unloaded OCR environment")
    if "paddle-ocr" not in [m["id"] for m in initial["inferswap"]["data"]]:raise RuntimeError("OCR not registered")
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in [*ROOT.glob("ocr-api/app/*.py"),ROOT/"Dockerfile",ROOT/"docker-compose.yml",
                        ROOT/"prepare-inferswap",ROOT/"configs/common.env",ROOT/"configs/models/paddleocr-vl-1.6.env"]}
    write(out/"integration_manifest.json",{"sources":sources,"paddle_sequence_budget":8,"qwen_request_budget":2,
        "wall_budget_seconds":900,"automatic_retries":0,"inference_gateway":INFER,
        "direct_control_post_calls":0,"fixture_scope":"four historical development crops; content quality is not accepted"})
    inspection=json.loads(command(["docker","inspect",CONTAINER]))[0]
    original_id=inspection["Id"];write(out/"container_before_stop.json",inspection)
    command(["docker","compose","stop","-t","45","paddle-ocr-runtime"])
    stopped=json.loads(command(["docker","inspect",CONTAINER]))[0];write(out/"container_stopped.json",stopped)
    if stopped["State"]["Running"]:raise RuntimeError("container stop did not complete")
    log_path=out/"inferswap_server.log"
    log_offset=log_path.stat().st_size
    stop=threading.Event();samples=[];phase={"name":"cold_prepare_load"};start=time.monotonic()
    def sample():
        count=0
        while not stop.is_set():
            row={"time":time.time(),"monotonic":time.monotonic(),"phase":phase["name"]}
            for name,base,path in [("ocr",OCR,"/control/status"),("qwen",QWEN,"/control/status"),("inferswap",INFER,"/v1/models")]:
                try:row[name]=get(base,path)
                except Exception as exc:row[name+"_error"]=repr(exc)
            if count%5==0:
                try:row["gpu_used_mib"]=int(command(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"]).strip().splitlines()[0])
                except Exception as exc:row["gpu_error"]=repr(exc)
            samples.append(row);count+=1;stop.wait(.2)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    summary={"state":"failed","direct_control_post_calls":0}
    try:
        for fixture in fixtures:
            phase["name"]="task_"+fixture["task"]
            row=ledger.reserve(fixture,phase["name"])
            body,headers=multipart(fixture)
            row["request_body_sha256"]=hashlib.sha256(body).hexdigest()
            row.update(post_infer("paddle-ocr",body,headers))
            write(out/"proxy"/f"{row['index']:03d}_response.json",row);write(out/"proxy/ledger.json",ledger.calls)
            if (row["http_status"]!=200 or row["body"].get("task")!=fixture["task"]
                    or row["body"].get("input",{}).get("sha256")!=fixture["sha256"]
                    or row["body"].get("finish_reason")!="stop"
                    or row["body"].get("execution",{}).get("sequences_started")!=1):
                raise RuntimeError("proxy inference/byte preservation failed: "+fixture["task"])
            print(json.dumps({"task":fixture["task"],"http_status":row["http_status"],"seconds":row["seconds"]}),flush=True)
        running=get(OCR,"/control/status")
        if running["state"]!="ready" or running["residency"]!="resident":raise RuntimeError("OCR not resident after inference")
        write(out/"ocr_resident.json",running)
        inspection=json.loads(command(["docker","inspect",CONTAINER]))[0];write(out/"container_after_prepare.json",inspection)
        if inspection["Id"]!=original_id or not inspection["State"]["Running"]:raise RuntimeError("prepare recreated/stopped container")
        phase["name"]="proxy_disconnect"
        fixture=next(f for f in fixtures if f["task"]=="table");row=ledger.reserve(fixture,"proxy_disconnect")
        body,headers=multipart(fixture);connection=http.client.HTTPConnection("127.0.0.1",8095,timeout=30)
        try:
            connection.request("POST","/infer?model=paddle-ocr",body=body,headers=headers)
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                if get(OCR,"/control/status")["active_requests"]==1:break
                time.sleep(.05)
            else:raise RuntimeError("proxy cancellation request never became active")
            time.sleep(.5);before_cancel=get(OCR,"/control/status")
            closed=time.monotonic();connection.close()
            cancellation_samples=[]
            while time.monotonic()-closed<30:
                state=get(OCR,"/control/status");cancellation_samples.append(state)
                if state["active_requests"]==0:break
                time.sleep(.1)
            else:raise RuntimeError("proxy cancellation did not drain")
            row.update(client_disconnected=True,before=before_cancel,after=state,seconds_to_idle=time.monotonic()-closed)
            write(out/"proxy_cancel.json",row);write(out/"proxy/ledger.json",ledger.calls)
            if state["state"]!="ready" or state["residency"]!="resident":raise RuntimeError("cancel unexpectedly unloaded model")
        finally:connection.close()
        # This different model cannot coexist with OCR at the measured reservation bounds.
        # The only trigger is an inference request; InferSwap chooses and unloads the victim.
        phase["name"]="qwen_switch_unloads_ocr"
        qwen_payload={"model":"qwen3.8-27b","messages":[{"role":"user","content":"확인이라고만 답해."}],
                      "max_tokens":32,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
        result=post_infer("qwen-vlm",json.dumps(qwen_payload,ensure_ascii=False).encode(),{"Content-Type":"application/json"})
        write(out/"qwen_switch_response.json",result)
        if result["http_status"]!=200:raise RuntimeError("Qwen switch inference failed")
        end=time.monotonic()+10
        while time.monotonic()<end:
            ocr=get(OCR,"/control/status");qwen=get(QWEN,"/control/status")
            if ocr["state"]=="unloaded" and ocr["residency"]=="not_resident" and ocr["active_requests"]==0 and qwen["state"]=="ready":break
            time.sleep(.2)
        else:raise RuntimeError("OCR unload/Qwen residency not confirmed")
        workers=worker_processes()
        if workers:raise RuntimeError("OCR worker remained after InferSwap unload")
        final_inspection=json.loads(command(["docker","inspect",CONTAINER]))[0]
        if final_inspection["Id"]!=original_id or not final_inspection["State"]["Running"]:
            raise RuntimeError("model unload stopped/recreated container")
        log=log_path.read_bytes()[log_offset:].decode(errors="replace")
        lines=[line for line in log.splitlines() if 'paddle-ocr' in line]
        if not any('unload confirmed' in line and 'paddle-ocr' in line for line in lines):
            raise RuntimeError("InferSwap unload log evidence missing")
        write(out/"inferswap_ocr_events.json",lines)
        summary.update(state="inferswap_load_infer_unload_passed",container_started_by_prepare=True,
            same_container_preserved=True,ocr_unloaded_by_scheduler=True,ocr_workers_after=workers,
            qwen_ready_restored=True,paddle_requests=len(ledger.calls),qwen_requests=1)
    except Exception as exc:
        summary["error"]=repr(exc);print("INTEGRATION ERROR",repr(exc),flush=True)
    finally:
        stop.set();thread.join(timeout=20)
        write(out/"integration_samples.json",samples)
        for name,base,path in [("ocr_after",OCR,"/control/status"),("qwen_after",QWEN,"/control/status"),("inferswap_after",INFER,"/v1/models")]:
            try:write(out/(name+".json"),get(base,path))
            except Exception as exc:write(out/(name+".json"),{"error":repr(exc)})
        p=subprocess.run(["docker","logs",CONTAINER],capture_output=True,text=True)
        (out/"ocr_server.log").write_text(p.stdout+p.stderr)
        summary["wall_seconds"]=time.monotonic()-start;write(out/"integration_summary.json",summary);print(json.dumps(summary),flush=True)
    if summary["state"]!="inferswap_load_infer_unload_passed":raise SystemExit(1)


if __name__=="__main__":main()
