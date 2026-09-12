# Kimi-K3 CuTeDSL MLA fixed split-KV experiment

This opt-in experiment splits long KV scans into more independent tasks, so
mixed-length batches can keep more SMs busy after short requests finish. It
reuses FlashInfer's existing main kernel and FP32 split-output/LSE reduction.
Performance and numerical correctness must still be checked on GB300.

## Enable on Decode

Keep the existing `cutedsl_mla`, EP32/DP8/attention-TP4 deployment and set:

```bash
export SGLANG_CUTEDSL_MLA_NUM_KV_SPLITS=4
# Start the existing SGLang Decode command.
```

- `0` (default, or unset): original FlashInfer heuristic and public API path.
- `1..32`: request that many fixed KV partitions. Start comparisons with
  `0, 1, 2, 4, 8`; `1` also checks the adapter against the original B128 path.
- Restart workers for every value. Changing the environment after startup does
  not update captured graph scalars or workspace pointers.
- Requires `flashinfer-python==0.6.17`, DCP size 1, no speculative decoding,
  ordinary one-query-token decode, and no skip-softmax. Other FlashInfer
  versions fail explicitly when the override is enabled; default `0` does not
  impose a version check.

The startup log reports requested splits, maximum batch capacity and reserved
workspace. A short context can have fewer nonempty partitions than requested.
Graph replay continues to use real per-request `seq_lens`; the context limit
is only a host planning bound, not a padded effective KV length.

## Workspace and implementation

`cutedsl_mla_splitkv.py` creates an isolated copy of FlashInfer's monolithic
Python wrapper with only its split/workspace planner replaced. It does not
change the installed package or its process-global functions. This temporary
private-API adapter is pinned to 0.6.17; replace it with a public split argument
when one becomes available.

Planning reserves the same M128-padded FP32 partial-output and LSE layout as
FlashInfer, even for K3's 24 local heads. For B128, Q1 and latent dimension 512:

| Splits | Required partial workspace |
|---:|---:|
| 1 | 0 |
| 2 | 64.125 MiB |
| 4 | 128.25 MiB |
| 8 | 256.5 MiB |

The backend retains its existing 150 MiB buffer when sufficient. Otherwise it
allocates a private larger buffer before warmup/capture, sized for the aligned
request-pool capacity (also covering eager batches). The original shared buffer
remains available to other backend instances, so the private allocation is
additional memory. No workspace growth or host read of GPU lengths occurs in
the decode layer loop.

## Single-GPU correctness and timing

Run on an idle GB300 GPU with the branch installed and FlashInfer 0.6.17:

```bash
CUDA_VISIBLE_DEVICES=0 python test/manual/bench_cutedsl_mla_splitkv.py \
  --batch-size 128 --heads 24 --splits 1 2 4 8 \
  --max-seq-len 1048576 --enable-pdl
```

The script compares the stock monolithic wrapper and the override on identical
Q, KV, page tables and length arrays. It uses randomly placed, disjoint pages
and checks finite outputs, numerical differences and graph replay after
changing device metadata. CUDA Graph timings include the main kernel and the
split reduction. Reported tolerances are experimental screening thresholds,
not a replacement for model accuracy validation.

Built-in cases keep B and query width constant:

- Uniform: every request has 64K context.
- Mixed, matched mean: 75% at 8K and 25% at 232K (64K mean for B128).
- Mixed, shorter mean: 87.5% at 8K and 12.5% at 256K (39K mean for B128).

Use `--help` for custom lengths, dtype, padding rows and timing controls. The
standalone script does not start a model or modify a running service.

## End-to-end A/B

Keep context-bucket routing and all other settings identical in both arms,
including `--load-snapshot-publish-interval 1`. Compare the same replay rather
than changing `random-range-ratio`, concurrency or request rate between arms.
Record effective B/graph B, length mean/P95/max, MLA main-plus-reduction time,
MegaMoE time, throughput and P99 ITL. Verify output quality before performance
rollout. Increasing split count can regress uniform/short workloads because
it adds intermediate writes and reduction work.

Rollback: unset `SGLANG_CUTEDSL_MLA_NUM_KV_SPLITS` and restart Decode workers.
