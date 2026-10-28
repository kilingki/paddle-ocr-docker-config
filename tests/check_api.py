import argparse
from pathlib import Path
from runtime_support import Ledger, load_fixtures, write


def check_api(base, fixtures, ledger, label="smoke"):
    results = [ledger.infer(base, fixture, label) for fixture in fixtures]
    for result in results:
        body = result.get("body",{})
        if (result["http_status"] != 200 or body.get("finish_reason") != "stop"
                or body.get("execution",{}).get("sequences_started") != 1):
            raise RuntimeError("single-task API check failed; preserved raw response")
    return results


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--base-url",default="http://127.0.0.1:8020")
    p.add_argument("--fixtures",default="tests/runtime_fixtures.json")
    p.add_argument("--out",required=True)
    args=p.parse_args()
    check_api(args.base_url,load_fixtures(args.fixtures),Ledger(Path(args.out)), "api")
