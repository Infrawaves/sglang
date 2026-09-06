# Decode TP hang diagnostics

This branch adds observations, not a proposed hang fix. It preserves sampling,
collectives, demotion, CUDA Graph, and stream ordering. It adds no CUDA sync,
tensor D2H transfer, extra collective, or GPU tensor value hash.

## Enable before startup

```bash
export SGLANG_DEBUG_DECODE_HANG=1
export SGLANG_DEBUG_DECODE_HANG_DIR=/model/decode-hang/ranks
export SGLANG_DEBUG_DECODE_HANG_MAX_MB=64
export SGLANG_DEBUG_DECODE_HANG_BACKUPS=4
```

Each scheduler writes a unique host/PID/start-time JSONL file, with approximately
64 MiB per segment and four rotated backups by default. Lines flush to the OS on
each event (no per-event fsync). Logging failure disables the writer and warns
once; files are a recent-history window, not an unlimited audit trail. Copy all
segments promptly on a hang. Do not write per-step logs directly to NFS.

`trace_start` includes Torch version, source path, and hashes of the logger,
sampler, and Decode source files. Preserve this record and the deployed image
digest/build commit alongside the logs. A path/tag alone does not prove version.

## Read the records

| Event / field | Meaning |
|---|---|
| `schedule_enter`, `schedule_return` | Ordered running/waiting/selected request rows around scheduling and filtering |
| `batch_submit`, `forward_iter`, `rids_hash` | Immutable batch identity at forward submission; SHA-256 preserves RID order |
| `graph_replay_submit` | Raw batch size and selected graph size; distinguish e.g. 62/63 real requests using the same padded graph |
| `delayed_sample_enter` | Original forward context restored after intervening overlap work |
| `token_sync_enter`, `token_sync_skip` | Actual MIN branch decision, tensor numel/shape/dtype/device, env/grammar condition |
| `sample_seq`, `min_seq` | Local sampler visits versus actual sampler MIN submissions; these are not NCCL opCount |
| `pg_name`, `pg_ranks`, `pg_seq` | PyTorch group identity/membership and host collective sequence immediately BEFORE this call; unsupported fields are null |
| `token_sync_host_return`, `batch_host_return` | CPU call returned; **not** GPU completion |
| `copy_done_wait_enter`, `copy_done_wait_return` | Before/after the existing completion-event synchronization, without adding a sync |
| `result_process_enter`, `result_process_return` | Request lengths, last committed CPU token, finish/retract/demote state around output processing |
| `transfer_admit`, `retract_*`, `demotion_*` | Admission, retraction, KV backup/restore and removal transitions |
| `poll_min_enter`, `poll_min_return` | CPU/Gloo poll-state MIN, distinct from GPU/NCCL token MIN |

Compact request rows use the order:
`[rid, output_len, last_token, finished_reason_type, is_retracted, is_demoted]`.
No prompt content is logged. Warmup calls outside Scheduler can lack a
`forward_iter`; do not align them with serving batches by wall clock alone.
`pg_seq` uses optional `_get_sequence_number_for_group()` and is a host submission
counter, not device progress or a sequence for CustomAR/other NCCL communicators.

Compare all ranks within the same replica and TP group. Find the last matching
batch/RID order/lengths and the first difference, then inspect the intervening
finish/filter/admit/retract/restore events. Same count is insufficient: different
RID order also means semantically different reductions. Different local forward
counters alone are not proof of the same collective's mismatch.

## Kernel and NCCL logging

```bash
mkdir -p /model/decode-hang
export SGLANG_KERNEL_API_LOGLEVEL=3
export SGLANG_KERNEL_API_LOG_INCLUDE='*.custom_all_reduce,k3.*'
export SGLANG_KERNEL_API_LOGDEST='/model/decode-hang/kernel-api-%i.log'
export SGLANG_KERNEL_API_LOG_MAX_MB=64
```

Level 3 records host API names and tensor metadata. Opaque non-tensor objects are
logged by type only at this level; their repr/properties may read GPU storage.
Enums retain their symbolic names (e.g. the CustomAR algorithm). The INCLUDE filter applies
to all log levels and leaves excluded callables unwrapped. Existing DUMP_INCLUDE
filters only tensor dumps. Files with MAX_MB > 0 retain four backups; 0 keeps the
original unbounded handler. `%i` is the PID; put files in a unique per-node/run
directory to prevent cross-node PID collisions.

Covered here: CustomAR's existing wrapper and five explicit K3 fused AR entries.
Under CUDA Graph these Python calls describe capture/eager execution, **not every
replay of each GPU kernel**. `graph_replay_submit` records actual replay submission.
The sampler's direct PyTorch MIN is covered by the structured trace, not CustomAR.
Do not use levels 5/10, CUDA_LAUNCH_BLOCKING, GPU printf, or disable CUDA Graph in
the baseline: they can change timing or add synchronization. Level 3 and file I/O
still have overhead; measure ITL and log rate on the reproduction instance.

NCCL INFO records initialization/configuration; do not assume it records every
collective. If needed for a short controlled run, set before startup:

```bash
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,COLL
export NCCL_DEBUG_FILE='/model/decode-hang/nccl-%h-%p.log'
```

TRACE can expose call/count/order information (version-dependent). Native NCCL
files do not rotate here; monitor local disk and limit the run. These logs do not
trace CustomAR kernel progress. No NCCL algorithm/protocol is forced.

## Small coredump and incident order

Recommended for this round:

```bash
mkdir -p /model/coredumps
export CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_FILE='/model/coredumps/cuda_%h_%p_%t'
export CUDA_COREDUMP_SHOW_PROGRESS=1
export CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory'
```

CUDA 13.1 documents that `skip_global_memory` also excludes constbank memory.
Shared/local memory are retained. Active kernels/PCs/registers and available
launch metadata can identify the current wait, but pointer-referenced token
buffers, CustomAR workspaces/semaphores, and global NCCL work queues may be absent.
Even a full dump is a snapshot and cannot independently reconstruct the first
historical divergence; correlate it with the flight log.

Before any dump trigger, collect all-rank CPU stacks and freeze copies of logs on
both nodes. Then coordinate both nodes' dump triggers as closely as possible;
record their times. Triggering interrupts GPU execution and can abort the target
(the supplied flags do not contain skip_abort). A dump-triggered peer pause must
not be mistaken for the original fault. Existing collect_decode_hang.sh --check
is a startup check, not proof that a later dump completed. After triggering, wait
for driver stderr completion and verify the dump with cuda-gdb; a collector wait
timeout or stable file size alone does not prove completeness. Preserve node-local
originals until shared/local copies are verified.

## Evidence boundary

A 63/62 count split violates the NCCL contract only when it is established to be
the same collective on the same group. The documented result is undefined
behavior (hang, crash, or corruption), not a guaranteed hang. Detokenizer timeout
is consistent with upstream inference stopping; use the incident timeline to
establish direction. This patch does not assert a CustomAR or demotion root cause.

## Local validation

CPU tests: `python3 test/registered/unit/observability/test_decode_hang.py`.
They cover overlap context, nested prebuilt batches, disabled logging, rotation,
I/O failures, sampler call conditions, and kernel logging filters. On the local Mac
these were run using a stdlib harness that bypasses package GPU imports and the
heavy test base; the real trace/env code and extracted leaf methods are executed.
TP8/GB300 execution and overhead still require a reproduction deployment.

The base commit already contains an unresolved `_make_abort_req` reference in the
demotion-backup-failure branch of decode.py (ruff F821). It is not introduced or
fixed by this logging patch; no existing evidence proves that branch was hit.

Sources: [NCCL collective contract](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html),
[CUDA 13.1 coredump flags](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-gdb/index.html#gpu-core-dump-support),
[PyTorch host sequence implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L1182).
