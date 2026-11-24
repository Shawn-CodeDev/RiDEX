import json
import os
from pathlib import Path
import sys
import time

import requests
from ridex.services import check_services

if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Scheduler allocation required")
target = sys.argv[1]
deadline = time.monotonic()+2700
with requests.Session() as session:
    session.trust_env = False
    while True:
        try:
            document = (dict(profile="hpsv2", urls={"hpsv2": os.environ["HPSV2_REWARD_URLS"].split(",")})
                        if target == "external" else json.loads(Path(target).read_text()))
            check_services(document, session)
            break
        except (FileNotFoundError, requests.RequestException):
            if time.monotonic() >= deadline:
                raise TimeoutError("HPS service not available")
            time.sleep(10)
print("Canonical HPSv2 reward service is healthy", flush=True)
