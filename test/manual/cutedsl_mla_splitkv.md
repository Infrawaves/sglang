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
- Requires `flashinfer-python==0.6.17` or `0.6.18`, DCP size 1, no speculative decoding,
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
private-API adapter accepts only 0.6.17 and 0.6.18; replace it with a public
split argument when one becomes available.

SGLang upstream commit `39e147443bfd750252892e1dc2e46af8439b0679` pins
FlashInfer 0.6.18 and moves DCP metadata helpers to `TRTLLMMLABackend`. The
fixed split override remains in the ordinary non-DCP decode hook; keep the
upstream metadata helpers and fallback argument forwarding when rebasing.
The two FlashInfer releases have identical monolithic decode wrappers and
split/workspace planning interfaces. This source compatibility check does
not replace GB300 numerical and performance validation after upgrading.

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

Run on an idle GB300 GPU with the branch installed and FlashInfer 0.6.17 or 0.6.18.
The benchmark compares each fixed split with **FlashInfer's stock heuristic
planner for the same batch and inputs**. The stock effective split can change
with batch size. This is stock parity, not an independent high-precision
reference or a model accuracy test.

Reproduce the B128 case, including timing:

```bash
CUDA_VISIBLE_DEVICES=0 python test/manual/bench_cutedsl_mla_splitkv.py \
  --batch-size 128 --heads 24 --splits 1 4 8 16 32 --dtype fp8 \
  --max-seq-len 1048576 --enable-pdl \
  --output-jsonl splitkv-b128.jsonl
```

Scan multiple batches first with `--check-only` to skip timing:

```bash
CUDA_VISIBLE_DEVICES=0 python test/manual/bench_cutedsl_mla_splitkv.py \
  --batch-sizes 1 2 4 8 16 32 64 128 --heads 24 \
  --splits 1 4 8 16 32 --dtype fp8 --max-seq-len 1048576 \
  --enable-pdl --check-only --output-jsonl splitkv-batch-sweep.jsonl
```

`--batch-size` and `--batch-sizes` are mutually exclusive. Fixed splits default
to `1 4 8 16 32`. Within each case, all candidates use identical Q, KV, page
tables and length arrays; pages are randomly placed and disjoint. Different
batches are separate test inputs, so their error values are not a controlled
comparison of batch size alone.

Each JSONL event identifies the case, candidate and comparison phase:

- `case_start` and `candidate_start` identify work before it runs, including a
  candidate that subsequently fails.
- `check` reports `reference_split`, `effective_split`, maximum/mean absolute
  error, RMSE, mismatch count/fraction, nonfinite counts and the worst absolute
  error index. Check phases cover stock repeatability, eager-versus-graph
  parity where applicable, initial inputs, changed metadata and restored
  metadata.
- `result` reports the candidate outcome and timing when eligible; `summary`
  reports the overall outcome. `--output-jsonl` saves the events as well as
  printing them.

The top-level `result.max_abs_error` describes the initial-input comparison
(`stock_repeat` for stock). Inspect `phase_checks` for every phase's errors.
A single `check.passed` only describes that comparison; use the final
`result.status` and `reference_valid` to determine whether a candidate passed.

Numerical mismatches continue to the remaining candidates and batches by
default, then the process exits with status 1. Use `--fail-fast` to stop at
the first failed check. CUDA/runtime exceptions still terminate the run.
Failed candidates are not timed. If the stock baseline fails its own checks,
results have `reference_valid: false`; comparisons against that baseline do
not establish a pass for any fixed split. Tolerances are experimental
screening thresholds, not proof of model-level correctness.

Built-in cases use these actual KV lengths before `--length-scale`:

- `uniform64k`: every request has 64K context.
- `mixedmean64k`: 75% at 8K and 25% at 232K (64K mean for B128).
- `shortermeanlongtail`: 87.5% at 8K and 12.5% at 256K (39K mean for B128).

`--max-seq-len` sets the planning capacity, **not the actual KV length**.
Small batches cannot preserve the stated mixture percentages; inspect the
reported actual lengths. To test a real 1M KV sequence, scale only the uniform
case (scaling the mixed cases by 16 would exceed the 1M bound):

```bash
CUDA_VISIBLE_DEVICES=0 python test/manual/bench_cutedsl_mla_splitkv.py \
  --batch-size 1 --heads 24 --splits 1 4 8 16 32 --dtype fp8 \
  --distributions uniform64k --length-scale 16 --max-seq-len 1048576 \
  --enable-pdl --check-only --output-jsonl splitkv-1m.jsonl
```

To isolate PDL or graph behavior after a failed case, keep the batch, case,
seed and splits unchanged. Starting from a graph + PDL failure, use these
two separate comparisons; each changes one execution setting:

```bash
# Graph without PDL; compare with the original --enable-pdl run.
CUDA_VISIBLE_DEVICES=0 python test/manual/bench_cutedsl_mla_splitkv.py \
  --batch-size 128 --heads 24 --splits 1 4 8 16 32 --dtype fp8 \
  --distributions uniform64k --max-seq-len 1048576 --check-only \
  --execution-mode graph --output-jsonl splitkv-b128-no-pdl.jsonl

# Eager with PDL; compare with the original graph + PDL run.
CUDA_VISIBLE_DEVICES=0 python test/manual/bench_cutedsl_mla_splitkv.py \
  --batch-size 128 --heads 24 --splits 1 4 8 16 32 --dtype fp8 \
  --distributions uniform64k --max-seq-len 1048576 --check-only \
  --enable-pdl --execution-mode eager --output-jsonl splitkv-b128-eager.jsonl
```

Graph mode is the default. Timing includes MLA and its split reduction using
repeated data without an L2 flush; it is not end-to-end ITL or a simulation of
long autoregressive generation. The standalone script does not start a model
or modify a running service. Use `--help` for custom lengths, padding rows and
timing controls.

## End-to-end A/B

Keep context-bucket routing and all other settings identical in both arms,
including `--load-snapshot-publish-interval 1`. Compare the same replay rather
than changing `random-range-ratio`, concurrency or request rate between arms.
Record effective B/graph B, length mean/P95/max, MLA main-plus-reduction time,
MegaMoE time, throughput and P99 ITL. Verify output quality before performance
rollout. Increasing split count can regress uniform/short workloads because
it adds intermediate writes and reduction work.

Rollback: unset `SGLANG_CUTEDSL_MLA_NUM_KV_SPLITS` and restart Decode workers.
