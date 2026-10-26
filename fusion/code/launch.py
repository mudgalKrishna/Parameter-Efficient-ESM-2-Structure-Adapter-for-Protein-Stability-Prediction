"""Background job queue for the GPU: runs jobs from $ESM2_WORK/queue.txt, N at a time."""
import subprocess
import sys
import time
from pathlib import Path

import paths

HERE = Path(__file__).resolve().parent

Q = paths.WORK / "queue.txt"
DONE = paths.WORK / "queue_done.txt"
LOG = paths.LOGS
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
        running[j] = subprocess.Popen([sys.executable, str(HERE / "stab.py"), *j.split()],
                                      stdout=fh, stderr=subprocess.STDOUT)
    time.sleep(10)
