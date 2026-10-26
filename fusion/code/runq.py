"""Run a queue of stab3.py jobs with N workers, skipping any already finished.

  python runq.py <queue_file> <n_workers>

Each queue line is the argument string after `python stab3.py`. A job counts as done when its
output .npz exists, so the queue is restartable: the launcher can be killed and re-run without
repeating finished work or trusting a separate bookkeeping file (round 2 lost 8 runs to a
done-file that recorded crashed jobs as complete).
"""
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import paths

W = paths.WORK
Q = Path(sys.argv[1])
NW = int(sys.argv[2]) if len(sys.argv) > 2 else 1

jobs = [l.strip() for l in Q.read_text().splitlines() if l.strip() and not l.startswith("#")]


def outfile(argstr):
    tag = re.search(r"--tag (\S+)", argstr).group(1)
    sp = re.search(r"--split (\S+)", argstr)
    sd = re.search(r"--seed (\S+)", argstr)
    return W / "runs3" / tag / f"{sp.group(1) if sp else 'mega'}_seed{sd.group(1) if sd else '0'}.npz"


todo = [j for j in jobs if not outfile(j).exists()]
print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run", flush=True)

lock = threading.Lock()
idx = [0]
log = W / "logs"
log.mkdir(exist_ok=True)


def worker(wid):
    while True:
        with lock:
            if idx[0] >= len(todo):
                return
            i = idx[0]
            idx[0] += 1
        j = todo[i]
        tag = re.search(r"--tag (\S+)", j).group(1)
        sp = re.search(r"--split (\S+)", j)
        sd = re.search(r"--seed (\S+)", j)
        name = f"{tag}_{sp.group(1) if sp else 'mega'}_s{sd.group(1) if sd else '0'}"
        t0 = time.time()
        print(f"[w{wid}] START {i + 1}/{len(todo)} {name}", flush=True)
        with open(log / f"{name}.log", "w") as fh:
            r = subprocess.run(["python", "-I", str(W / "stab3.py")] + j.split(),
                               stdout=fh, stderr=subprocess.STDOUT)
        ok = outfile(j).exists()
        print(f"[w{wid}] {'OK  ' if ok else 'FAIL'} {name} rc={r.returncode} "
              f"{time.time() - t0:.0f}s", flush=True)


ths = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(NW)]
for t in ths:
    t.start()
for t in ths:
    t.join()
print("QUEUE COMPLETE", flush=True)
