# DyF — Dynamic Floor

Element-wise replacement for LayerNorm in pre-norm Transformers / ConvNeXt:

```
DyF(x)_k = γ_k · [ψ(x_k − a) + a] + β_k

           SiLU:  ψ(v) = v · σ(v)
           GELU:  ψ(v) = v · Φ(v)
           Hard:  ψ(v) = max(0, v)        →  DyF reduces to γ_k·max(a, x_k) + β_k
```

Per-module **scalar** `a` (saturation floor); per-channel `γ`, `β`.
Total params per module: **2C + 1**.

This repo extends the [DyT](https://github.com/jiachenzhu/DyT) codebase
with `dynamic_floor.py` and a NELU-style training orchestrator that
runs on AWS spot via either dstack or bare-EC2.

For the full mathematical derivation see `docs/spec.md`. The upstream
DyT README is preserved as `README_DyT_upstream.md`.

---

## Why DyF

| | LN | DyT | DyF |
|---|---|---|---|
| (P1) Element-wise (no token statistics)            | ✗ | ✓ | ✓ |
| (P2) Positive unbounded — outliers preserved        | ✗ | ✗ | ✓ |
| (P3) Negative bounded — saturates                   | ~ | ✓ | ✓ |
| (P4) Affine DOF at inference                        | 2C | 2C | **2C+1** |
| (P5) Includes shifted ReLU & identity               | ✗ | ✗ | ✓ |

The asymmetry (positive unbounded, negative bounded) plus a single
learned floor `a` is the core differentiator from LN/DyT. Two soft
kernels (SiLU, GELU) and one hard kernel (ReLU) are supported as the ψ
choice.

---

## Predictions to validate

| ID  | Prediction | Experiment |
|-----|------------|------------|
| C1  | DyF ≥ LN, DyT on **pre-norm** Transformer / ConvNeXt | `vit_base.yaml × {ln,dyt,dyf-silu,dyf-gelu,dyf-hard}` |
| C2  | DyF degrades or diverges on **post-norm** | post-norm fork (TODO; see §"Future") |
| C3  | Learned `a` ↔ channel-wise lower-tail of inputs | `dyf-{silu,gelu}-{aneg,apos}` ablation |
| C4  | Per-block `a_init` matters at LLM scale | LLM follow-up, out of scope here |
| C5  | ViT-B / ImageNet-1K: within ±0.5%p of DyT, ±0.3%p of LN | `vit_base.yaml` rows |

The default job queue (`scripts/infra/default_job_order.txt`) covers
C1, C3, C5 with 14 runs.

---

## Repository layout

```
dynamic_tanh.py              # upstream DyT
dynamic_floor.py             # ← DyF: DynamicFloor + convert_ln_to_dyf
main.py                      # upstream trainer + DyF flags + SIGTERM handler
configs/
  _base.yaml                 # shared defaults (optimizer, aug, AMP, seed)
  imagenet/
    vit_base.yaml
    convnext_tiny.yaml
    convnext_small.yaml
  dryrun/toy.yaml            # 2-epoch CPU smoke test
scripts/
  orchestrate.sh             # WORKER: S3 lease + sentinel + torchrun loop
  bootstrap.sh               # bare-EC2 boot: mount /data, clone repo, hand off
  dryrun.sh                  # local CPU smoke test
  launch_local.sh            # local: spawn N workers (background, no AWS)
  launch_workers.sh          # cloud: dstack apply N copies of worker.dstack.yml
  launch_campaign.sh         # cloud: bare-EC2, spawns watchdog
  infra/
    activation_flags.sh      # variant name → main.py CLI flags
    yaml_to_args.py          # YAML (with include:) → argparse-style flags
    default_job_order.txt    # the queue (one <cfg>:<act> per line)
    setup_aws.sh             # one-shot: bucket + IAM role + instance profile
    trust-policy.json        # IAM trust policy for dyf-worker-role
    worker-policy.json       # IAM permissions (S3 + EC2 lifecycle)
    user-data.sh             # template for EC2 user-data (placeholders)
    render_user_data.sh      # substitute @@VAR@@ → values from .env
    run_worker.sh            # aws ec2 run-instances for one spot worker
    watchdog.sh              # keep TARGET_WORKERS alive until queue drains
.dstack/
  profiles.yml               # spot/region/AZ/retry settings
  worker.dstack.yml          # per-worker dstack task definition
.env.example                 # copy → .env, fill in
.dstackignore                # excluded paths from dstack workdir sync
checkpoints/, logs/          # local runs (created on demand)
```

---

## Three ways to run

### A) Local CPU smoke test (no AWS, no GPU)

```bash
bash scripts/dryrun.sh
```

Generates a fake ImageFolder under `/tmp/dyf_dryrun_data` and runs 2
epochs through `main.py` for `ln`, `dyt`, `dyf-silu`, `dyf-gelu`,
`dyf-hard`. If all variants finish, the swap and train loop are wired
correctly. Restrict to one variant: `bash scripts/dryrun.sh dyf-silu`.

### B) Local single-host, multi-GPU (no AWS)

```bash
# Edit configs/_base.yaml:data_path to your ImageNet root, then:
bash scripts/launch_local.sh 2          # 2 parallel local workers
tail -f logs/worker-1.log
```

Workers write to `./checkpoints/<exp>/`. Lease lives on the local
filesystem — multi-host needs the cloud path below.

### C) AWS spot, multi-host (this is what you want for the real campaign)

The full path: **S3 lease + sentinel** for cross-host coordination,
**dstack** *or* **bare-EC2 + watchdog** for VM lifecycle.

Pick one of the two cloud sub-paths below — both share the orchestrator
and queue, they only differ in how spot VMs get launched.

---

## AWS first-time setup

You only need to do this once per AWS account.

### 1. Install + configure the AWS CLI

```bash
# macOS
brew install awscli

# Verify auth
aws sts get-caller-identity
```

You need either a personal IAM user with admin perms (for the one-shot
setup below) or the equivalent SSO role.

### 2. Set up `.env`

```bash
cp .env.example .env
# Open .env and fill in:
#   WANDB_API_KEY (or leave blank to disable W&B)
#   CKPT_BUCKET   (e.g. s3://dyf-checkpoints — pick a globally-unique name)
#   AWS_DEFAULT_REGION (us-west-2 recommended for H100 spot)
source .env
```

### 3. Run the AWS bootstrap

```bash
bash scripts/infra/setup_aws.sh
```

This creates (idempotently):
- the S3 bucket `$CKPT_BUCKET`
- IAM role `dyf-worker-role` with the policies in
  `scripts/infra/{trust,worker}-policy.json`
- instance profile `dyf-worker-profile` with that role attached

### 4. Decide how to give workers the dataset

Three options, in order of operational simplicity:

**(a) Already on each VM.** If you use a custom AMI with ImageNet baked
in at `/data`, no extra step. Set `_base.yaml:data_path: /data/imagenet`.

**(b) dstack volume.** Create a dstack-managed EBS volume, populate it
once, then reference it from `.dstack/worker.dstack.yml` under
`volumes:`. Recommended for the dstack path.

**(c) EBS snapshot + bare-EC2.** Make an EBS snapshot of a populated
data disk, set `DATA_SNAPSHOT=snap-...` in `.env`. `run_worker.sh` will
attach a fresh copy as `/dev/sdg` and `bootstrap.sh` mounts it at
`/data` read-only. Recommended for the bare-EC2 path.

ImageNet at full res is ~150 GB; snapshot/volume size of 500 GB leaves
headroom for unpacking and a few epochs of decoded cache.

---

## Cloud sub-path 1: dstack

Easiest if you already use dstack for other GPU work.

```bash
# Pre-reqs: pip install dstack[aws], dstack server running, AWS backend configured
source .env
bash scripts/launch_workers.sh 2          # spawn 2 workers
```

Each worker:
1. dstack provisions an H100:8 spot VM (per `.dstack/profiles.yml`).
2. NGC PyTorch container starts; pip-installs timm/wandb/pyyaml/awscli.
3. `orchestrate.sh` enters the S3 lease loop and trains until the queue
   drains or the spot is preempted.
4. dstack's `retry: { on_events: [interruption, ...] }` reschedules on
   another AZ; the new VM picks up the same job from the latest
   checkpoint via S3 sync + `auto_resume: true`.

Useful follow-ups:

```bash
dstack ps                                 # live runs
dstack logs dyf-worker-1 -f               # follow one
aws s3 ls $CKPT_BUCKET/ --recursive | grep complete
```

## Cloud sub-path 2: bare-EC2 + watchdog

No dstack dependency — `aws ec2 run-instances` direct + a watchdog
process keeps the fleet alive.

You need extra `.env` vars for this path:

```bash
# Required for run_worker.sh
AMI=ami-0027d9a89a2d7f75b              # AWS DL Base AMI for your region
SG=sg-xxxxxxxxxxxxx                    # security group with ssh-22 allowed
KEY=YourKeyName                        # EC2 key pair (omit if you don't need SSH)
DATA_SNAPSHOT=snap-xxxxxxxxxxxxx       # ImageNet snapshot (option (c) above)
SUBNET_us_west_2a=subnet-xxx
SUBNET_us_west_2b=subnet-yyy
SUBNET_us_west_2c=subnet-zzz
SUBNET_us_west_2d=subnet-www

# REPO_URL/REPO_REF for bootstrap.sh's git clone
REPO_URL=https://github.com/<you>/DyF.git
REPO_REF=main
```

Then:

```bash
source .env
bash scripts/launch_campaign.sh           # foreground watchdog; Ctrl-C to stop
```

The watchdog:
- counts EC2 instances tagged `Project=dyf,Role=worker,Campaign=$CAMPAIGN`
- launches more (rotating through `CAMPAIGN_AZS`) until `TARGET_WORKERS`
- exits 0 once every job has a `complete` sentinel in S3
- never escalates to on-demand if all AZs reject the spot request —
  sleeps `CAPACITY_SLEEP_SEC` and retries the AZ cycle, forever

---

## What gets stored in S3 per experiment

Layout under `$CKPT_BUCKET/<exp>/` (where `<exp> = <config_basename>-<activation>`):

| File | Meaning |
|------|---------|
| `lease`              | "<owner-id> <unix-ts>". Refreshed every `HEARTBEAT_EVERY` (60s). Stale > `LEASE_TTL` (600s) → can be stolen. |
| `complete`           | Sentinel: training finished cleanly. Watchdog stops counting it once present. |
| `log.txt`            | `main.py` stdout + stderr (appended). |
| `checkpoint-*.pth`   | Per-epoch checkpoints (rolling, capped at `save_ckpt_num=3`). |
| `wandb_run_id.json`  | W&B run id sidecar. Stolen-lease workers re-attach to the same W&B run. |

A worker's resume protocol:
1. `aws s3 sync $CKPT_BUCKET/<exp>/ /tmp/runs/<exp>/`
2. main.py's `auto_resume: true` finds `checkpoint-*.pth`, loads it.
3. `WandbLogger.__init__` finds `wandb_run_id.json`, reuses the run id.
4. SIGTERM (from spot preempt watcher) → finish current epoch → exit.
5. heartbeat sync pushes the latest checkpoint; lease is released.

---

## Adding a new normalization variant

1. Add a case to `scripts/infra/activation_flags.sh`:
   ```bash
   dyf-bigA)   printf '%s' "--dynamic_floor true --dyf_kernel silu --dyf_a_init 2.0" ;;
   ```
2. Add rows to `scripts/infra/default_job_order.txt`, or pass via
   `JOB_ORDER="..." bash scripts/launch_workers.sh 2`.

The variant name flows into the experiment dir as `<config>-<variant>/`,
so it must be filesystem-safe (no `/`, spaces, colons).

## Adding a new model

1. Drop a YAML under `configs/imagenet/` that
   `include: ../_base.yaml` and overrides `model:`, `batch_size:`, etc.
2. `main.py` only knows `convnext*` and `vit*` model families today.
   For other families, extend the `if "convnext" in args.model:` branch.
3. Reference the new YAML in `default_job_order.txt`.

---

## Cost back-of-envelope

| Resource | Spot price (us-west-2, late-2025) | Notes |
|---|---|---|
| `p5.48xlarge` (8×H100) | ~$15-20/hr | usually cheapest in `us-west-2d` |
| `g5.12xlarge` (4×A10G) | ~$2-3/hr   | for dryrun / smaller models |
| S3 (CKPT_BUCKET)       | $0.023/GB·mo + ~$0.005/1000 PUTs | <$5/mo for the campaign |
| EBS (data volume)      | $0.08/GB·mo gp3 + $0.005/IOPS·mo | <$50/mo for 500 GB @ 16k IOPS |

ViT-B / ImageNet-1k @ 300 epochs ≈ 16-22 h on 8×H100 → ~$300/run.
The default 14-run queue is roughly **$3-5k**; cut to a 4-run subset
for first-pass C1 validation: ~**$1k**.

---

## Future / not yet implemented

- **(C2) Post-norm comparison.** DyT's `main.py` only swaps LN inside a
  pre-norm model. C2 needs a custom post-norm ViT (or borrowing the
  swap from a post-norm reference impl). Tracked.
- **Per-channel `a` ablation.** The spec discusses both scalar and
  per-channel `a`; current code uses scalar. Switching is a one-line
  change in `dynamic_floor.py:DynamicFloor.__init__`.
- **Diagnostic logging.** NELU logs per-epoch γ stats / weight norms.
  Adding `dyf_stats(model)` (mean/std of `(γ, β)`, value of `a`) would
  directly support C3 evaluation without offline analysis.
- **Eval orchestrator.** NELU has `eval_orchestrate.sh` for downstream
  robustness eval; DyF doesn't yet — once C1/C5 finish, port it.
