# Running this on hosted GPUs, without tripping anyone's rules

The training sweeps here are long batch jobs. Hosted notebook services differ sharply in whether
that is allowed, and I ran into both of the problems below the hard way. Both are avoidable.

## 1. Pick a host that permits batch jobs

Interactive notebook services — molab among them — generally state that notebooks are for
*interactive* compute, and reserve the right to terminate notebooks used for non-interactive
jobs. Detaching a training run with `setsid`/`nohup` so it survives a disconnect is precisely
what that prohibits, even though it is the obvious way to protect a long run from a dropped
connection.

| Host | Long batch sweeps |
|---|---|
| Kaggle Notebooks | yes — 9-hour sessions, ~30 GPU-hours/week, jobs continue when the tab closes |
| Google Colab | yes, within session limits |
| A rented VM / local GPU | yes |
| Interactive-only notebook services | **no** — run short cells there, not sweeps |

Nothing in this repository detaches a process. `launch2.py` and `runq.py` run their jobs as
ordinary foreground subprocesses of the launcher, so the work stops when you stop the launcher.
Keep it that way: if a run needs to survive a disconnect, move it to a host that allows that
rather than detaching it.

**Do not** write a watchdog that relaunches work after a host stops it. Repeatedly restarting
something a provider has terminated reads as circumventing enforcement, which is usually a
bannable offence rather than a terminable one.

## 2. Avoid command-line flags that look like a crypto miner

`rl_post.py` originally took `--algo` to select between REINFORCE, GRPO and so on. Mining
software is launched the same way — `xmrig --algo rx/0`, `t-rex --algo ethash` — so an abuse
detector scanning process command lines flagged a GPU-saturating process whose arguments matched
a miner, and suspended the account. A false positive, but a predictable one.

The flag is now `--method`. When naming CLI arguments for GPU-heavy jobs, it is worth steering
clear of the miner vocabulary generally: `--algo`, `--pool`, `--wallet`, `--worker`, `--rig`,
`--intensity`, `--stratum`, `--donate`.

## 3. What this code actually does

For anyone reviewing it after an automated flag:

- Trains protein stability predictors (ESM-2 + ProteinMPNN) on public research datasets:
  Tsuboyama et al. mega-scale ΔΔG, FireProtDB, S669.
- `setup_env.py` downloads model checkpoints and datasets over HTTPS from GitHub, Zenodo and
  RCSB. It writes them to disk and never executes anything it downloads.
- `run_structure_track.py` and `extra_rank32.py` call `exec()` on function bodies parsed with
  `ast` from this repository's own marimo notebook (`train_esm2_spurs_fixed.py`). marimo stores
  cells as functions, and these scripts drive that notebook as a plain script. The source is a
  fixed local path; nothing is fetched, decoded or executed from the network.
- No sockets are opened, no services are exposed, no `shell=True`, no `os.system`.
