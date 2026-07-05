# ARCC request — `highgpu` partition access on Newton

**Why:** our account (`skattel`, user `sa305415`) is associated only with `normal, ood, preemptable`, whose
H100 nodes expose **2 GPUs each**. The 8×H100 nodes (`evc101–104`) live in the `highgpu` partition, which our
account **cannot submit to** — `srun` returns *"Invalid account or account/partition combination."* We need a
single 8×H100 node for a ~135M language-model training run.

**Action needed:** the **PI must send this** (only the PI can request an association change for the account).

---

**To:** arcc-request@ucf.edu
**Subject:** Request `highgpu` partition access for account `skattel` (Newton)

Hello ARCC team,

Could you please add submit access to the **`highgpu`** partition (nodes `evc101–104`) on **Newton** for our
Slurm account **`skattel`** (`arcc_pi_skattel`), user **`sa305415`**? The account is currently associated only
with `normal`, `ood`, and `preemptable`, and submissions to `highgpu` fail with *"Invalid account or
account/partition combination."*

We have a language-model training project that needs a single 8×H100 node; the 2-GPU H100 nodes in `normal`
are the current limit for us. A standard `highgpu` association (or the appropriate QOS) would let us run.

Thank you,
[PI name] — `arcc_pi_skattel`

---

*Once granted, no code change is needed: the trainer is single-node DDP and launches on 8×H100 with
`--gres=gpu:8` / `--nproc_per_node=8`.*
