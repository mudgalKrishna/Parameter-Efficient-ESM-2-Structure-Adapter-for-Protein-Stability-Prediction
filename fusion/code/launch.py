"""Background job queue on the sandbox GPU: runs jobs from /marimo/work/queue.txt, N at a time."""
import subprocess
import sys
import time
from pathlib import Path

Q = Path("/marimo/work/queue.txt")
DONE = Path("/marimo/work/queue_done.txt")
LOG = Path("/marimo/work/logs")
LOG.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
running = {}
started = set(DONE.read_text().splitlines()) if DONE.exists() else set()
while True:
    jobs = [l.strip() for l in Q.read_text().splitlines() if l.strip() and not l.startswith("#")] if Q.exists() else []
    for k, pr in list(running.items()):
        if pr.poll() is not None:
            with DONE.open("a") as f:
                f.write(f"{k}\n")
            del running[k]
    for j in jobs:
        if len(running) >= N:
            break
        if j in started:
            continue
        started.add(j)
        name = j.split("--tag ")[1].split()[0] + "_s" + j.split("--seed ")[1].split()[0]
        fh = open(LOG / f"{name}.log", "w")
        running[j] = subprocess.Popen(f"cd /marimo/work && python stab.py {j}", shell=True, stdout=fh, stderr=subprocess.STDOUT)
    time.sleep(10)
