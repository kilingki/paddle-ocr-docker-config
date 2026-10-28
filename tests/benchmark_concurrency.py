import argparse, concurrent.futures, statistics, time
from pathlib import Path
from runtime_support import Ledger, load_fixtures, write


def benchmark(base, fixtures, ledger, concurrency, rounds=2):
    windows=[]
    for repeat in range(rounds):
        start=time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures=[pool.submit(ledger.infer,base,fixture,f"c{concurrency}_r{repeat}") for fixture in fixtures]
            results=[future.result() for future in futures]
        end=time.monotonic()
        if any(r["http_status"]!=200 or r["body"].get("finish_reason")!="stop" for r in results):
            raise RuntimeError("concurrency check failed/incomplete; no retry")
        windows.append({"concurrency":concurrency,"repeat":repeat,"start":start,"end":end,
            "seconds":end-start,"requests_per_second":len(results)/(end-start),
            "completion_tokens_per_second":sum(r["body"]["usage"]["completion_tokens"] for r in results)/(end-start),
            "indices":[r["index"] for r in results],
            "latencies_seconds":[r["seconds"] for r in results]})
    return windows


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--base-url",default="http://127.0.0.1:8020")
    p.add_argument("--fixtures",default="tests/runtime_fixtures.json")
    p.add_argument("--concurrency",type=int,choices=[1,2,4],required=True)
    p.add_argument("--rounds",type=int,default=2)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    out=Path(args.out);ledger=Ledger(out)
    write(out/"concurrency.json",benchmark(args.base_url,load_fixtures(args.fixtures),ledger,args.concurrency,args.rounds))
