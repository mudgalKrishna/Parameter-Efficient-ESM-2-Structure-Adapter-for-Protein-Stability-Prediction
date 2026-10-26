"""Job queue for stab2.py: runs lines of queue2.txt, N at a time."""
import subprocess
import sys
import time
from pathlib import Path

import paths

HERE = Path(__file__).resolve().parent

Q = paths.WORK / "queue2.txt"
DONE = paths.WORK / "queue2_done.txt"
LOG = paths.WORK / "logs2"
LOG.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
running, started = {}, set(DONE.read_text().split("\n")) if DONE.exists() else set()
while True:
    jobs = [l.strip() for l in Q.read_text().splitlines() if l.strip() and not l.startswith("#")] if Q.exists() else []
    for k, pr in list(running.items()):
        if pr.poll() is not None:
            with DONE.open("a") as f:
                f.write(k + "\n")
            del running[k]
    for j in jobs:
        if len(running) >= N:
            break
        if j in started:
            continue
        started.add(j)
        tag = j.split("--tag ")[1].split()[0]
        seed = j.split("--seed ")[1].split()[0]
        sp = j.split("--split ")[1].split()[0] if "--split " in j else "mega"
        fh = open(LOG / f"{tag}_{sp}_s{seed}.log", "w")
        running[j] = subprocess.Popen([sys.executable, str(HERE / "stab2.py"), *j.split()],
                                      stdout=fh, stderr=subprocess.STDOUT)
    time.sleep(10)
