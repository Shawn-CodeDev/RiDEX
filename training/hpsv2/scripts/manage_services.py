import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import requests
from ridex.services import check_services

if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Scheduler allocation required")
devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
if len(devices) != 1 or not devices[0].strip():
    raise ValueError("Allocate exactly one HPS service GPU")
ready = Path(sys.argv[1])
if ready.exists():
    raise FileExistsError(ready)
ready.parent.mkdir(parents=True, exist_ok=True)
port = 30000+int(os.environ["SLURM_JOB_ID"])%10000
document = dict(profile="hpsv2", server_job_id=os.environ["SLURM_JOB_ID"],
                urls={"hpsv2": [f"http://{socket.gethostname()}:{port}"]})
def stop(signum, frame):
    raise SystemExit(128+signum)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
child = None
with (ready.parent/"hpsv2.log").open("w") as log:
    try:
        child = subprocess.Popen(["bash", "scripts/server.sh", str(port)],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=devices[0]), stdout=log, stderr=subprocess.STDOUT)
        with requests.Session() as session:
            session.trust_env = False
            deadline = time.monotonic()+2700
            while True:
                if child.poll() is not None:
                    raise RuntimeError("HPS reward server exited; inspect hpsv2.log")
                try:
                    check_services(document, session)
                    break
                except requests.RequestException:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("HPS service startup timed out")
                    time.sleep(10)
        temporary = ready.with_suffix(".tmp")
        temporary.write_text(json.dumps(document, indent=2)+"\n")
        temporary.replace(ready)
        print(document, flush=True)
        while child.poll() is None:
            time.sleep(10)
        raise RuntimeError("HPS service exited")
    finally:
        ready.unlink(missing_ok=True)
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
