Draft follows the [official vLLM performance issue template](https://github.com/vllm-project/vllm/blob/main/.github/ISSUE_TEMPLATE/700-performance-discussion.yml). Replace the environment placeholders before submitting.

**Title:** `[Performance]: Reduce load_weights synchronization overhead for DeepSeek V4 Flash weight updates on H200`

### Proposal to improve performance

Profiling repeated weight updates for **DeepSeek V4 Flash on NVIDIA H200** revealed substantial CPU waiting and accounting overhead inside `load_weights`.

For the second `load_weights` invocation in each trace, the combined experimental optimizations reduced profiled CPU duration from **1,923.691 ms to 276.823 ms**, an **85.61% reduction / 6.95× speedup**.

The associated PR implements the **CPU expert-map cache** described below. The overall result includes three optimizations; separate end-to-end measurements for each optimization are not yet available.

**Original loading flow**

The baseline follows the existing layerwise reload path:

1. `load_weights` routes incoming weight names to parameters, shards, and experts.
2. `online_process_loader` buffers weights and calls `get_numel_loaded` to track layer completion. Accounting executes the original loader against meta parameters under `CopyCounter`.
3. Once a layer is ready, `_layerwise_process` materializes its parameters and replays the buffered loaders for actual loading.
4. Quantization post-processing prepares MXFP4 MoE weights for Marlin: repack `w13` and `w2`, then permute and encode their scales per expert.
5. Processed values are copied into the existing kernel tensor storage.

Accounting and actual loading are interleaved as layers become ready. Expert ID mapping occurs in both paths.

**Bottlenecks and implemented optimizations**

1. **GPU scalar reads during expert ID mapping**

   The baseline reads `expert_map[global_id].item()` from a GPU tensor for host-side decisions. Across **3,048 lookups**, the trace records **3,048 D2H transfers and 3,048 stream synchronizations**.

   Mapping consumes **959.661 ms**, including **870.633 ms** in CPU stream synchronization, while the corresponding D2H transfers total only **7.177 ms** of GPU time. This identifies frequent host waiting as the main cost.

   **Optimization:** cache a Python list whenever the expert map is created or recalculated, and use it for host-side ID and locality queries.

   **Observed result:** mapping duration falls to **0.287 ms**. Lookup count remains **3,048**, while D2H transfers and stream synchronizations inside these lookups fall to **zero**.

2. **Replaying expert loaders for loaded-element accounting**

   The baseline invokes `RoutedExperts.weight_loader` both for accounting and for actual loading: **3,048 calls**, including **1,524 accounting calls**.

   **Optimization:** use a counting fast path that determines the loaded-element count without re-entering the expert loader.

   **Observed result:** total expert-loader calls fall to **1,524**, all on the actual loading path. `CopyCounter.__torch_dispatch__` invocations fall from **4,936 to 96**.

   The fast path still checks expert placement, so expert-map lookup count remains unchanged.

3. **Synchronization in Marlin scales preparation**

   The baseline PyTorch scales path performs permutation and MXFP4 encoding per expert. These operations account for **256 H2D transfers and 256 stream synchronizations**, with **594.433 ms** spent in CPU stream synchronization.

   **Optimization:** implement scale permutation and encoding with CUDA operators.

   **Observed result:** the two scales-preparation entries fall from **615.190 ms to 7.704 ms**, and their H2D transfers and stream synchronizations fall to **zero**.

   Permutation and encoding still each execute **128 times** in this experiment. Scales-related GPU kernel time improves from **1.983 ms to 1.230 ms (1.61×)**; the much larger CPU-duration improvement reflects reduced host waiting and dispatch overhead.

### Misc discussion on performance

**Performance comparison**

The following breakdown assigns non-overlapping intervals within the selected `load_weights` invocation. Nested mapping time is removed from accounting and loader rows, and scale substeps are separated from their parent entries.

| Step | Baseline CPU duration (ms) | Optimized CPU duration (ms) | Baseline share | Optimized share |
| --- | ---: | ---: | ---: | ---: |
| Expert ID mapping | 959.661 | 0.287 | 49.89% | 0.10% |
| Loaded-element accounting, excluding mapping | 94.848 | 24.382 | 4.93% | 8.81% |
| Actual loader work, excluding mapping/accounting | 18.955 | 17.966 | 0.99% | 6.49% |
| Scales permutation | 304.153 | 2.591 | 15.81% | 0.94% |
| MXFP4 scales encoding | 308.717 | 3.183 | 16.05% | 1.15% |
| Other scales-entry work | 2.320 | 1.931 | 0.12% | 0.70% |
| Weight repack | 6.593 | 6.613 | 0.34% | 2.39% |
| Other work / gaps | 228.444 | 219.871 | 11.88% | 79.43% |
| **Total** | **1,923.691** | **276.823** | **100%** | **100%** |

Rounded rows may differ slightly from the total.

Expert mapping and scales preparation account for approximately **81.87%** of the baseline window. Actual loader work excluding mapping and accounting changes only modestly, from **18.955 ms to 17.966 ms**.

**Measurement scope**

- This compares exactly the **second chronological invocation** in each trace (`main` and `opt4`), using one observation per configuration.
- The window includes accounting, actual loaders, Marlin preparation, and CPU waiting. It excludes outer broadcast, bucket reconstruction, final synchronization/finalization, GC, and `empty_cache`.
- These are **profiled inclusive CPU durations**. GPU work can remain outstanding when the function returns.
- Matching operation counts do not establish identical tensor inputs or numerical correctness. The optimized full-update trace broadcasts approximately **0.0400% more bytes**.
- Repeated measurements with identical inputs and profiling disabled, plus CUDA numerical parity and model-output validation, are needed to establish the end-to-end benefit.

Related discussion: [#54477 — Direct Weight Reload for RL Training](https://github.com/vllm-project/vllm/issues/54477). This report contributes measurements and targeted optimizations within the existing layerwise reload path.

AI assistance was used to review the code and draft this issue.

### Your current environment (if you think it is necessary)

```text
Model: DeepSeek V4 Flash
GPU: NVIDIA H200
Profiled rank: 0
World size: 4
Profiled MoE preparation: MXFP4 → Marlin
CUDA runtime: 13.0
NCCL: 2.28.9
Profiler setting: with_stack=True

Baseline vLLM commit: <fill in>
Optimized vLLM commit: <fill in>
TP / EP / PP configuration: <fill in>
Weight-update entry point / reproduction command: <fill in>
Full collect_env.py output: <attach>
Profiling traces and analysis artifacts: <attach or link>
```

### Before submitting a new issue

- [ ] I have searched related issues and consulted the documentation chatbot.
