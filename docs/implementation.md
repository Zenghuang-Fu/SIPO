# SIPO implementation guide

## Execution path

`scripts/train.py` composes `configs/sipo.yaml` and launches
`verl.trainer.main_ppo`. The main entry initializes Ray workers and the
`RayPPOTrainer`. The FSDP worker selects the tree rollout implementation when
`actor_rollout_ref.rollout.mode=sync_with_tool_tree`.

`vLLMRolloutWithTools.generate_sequences` builds initial trajectories, executes
search tools, selects expansion points, creates additional continuations, and
scores terminal answers. It then computes corrected node values and token
advantages. `DataParallelPPOActor` consumes those tensors and applies the
configured `gspo_turn` policy loss. The trainer handles validation, saving,
and restoring checkpoints.

## SIPO components

| Component | Main code location | Configuration |
| --- | --- | --- |
| SFC | `ToolTreeNode.sample_expansion_nodes` | `expansion_score_norm`, `expansion_penalty_lambda`, `expansion_penalty_per_branch` |
| Branch expansion | `ToolTreeNode.create_branch` and the expansion loop in `generate_sequences` | `initial_rollouts`, `expansion_iterations`, `beam_size`, `num_branches` |
| OSC | `_expected_order_stat`, `_candidate_level_a2`, and the correction block in `generate_sequences` | `postsel_correction_weight` |
| Node/token credit | `ToolTreeNode.compute_value_from_children` and the final token-assignment loop | `leaf_value_norm`, `node_value_mode`, `node_adv_mode` |

These symbols are in
`verl/workers/rollout/vllm_rollout/vllm_rollout_with_tools_tree_offline.py`.
The complete rollout class is included, together with its decoding, tool,
reward, and distributed-training dependencies.

The default expansion score uses mean sampled-token surprisal. SFC standardizes
candidate scores and accounts for existing siblings. Fresh branches retain the
shared prefix and resume generation at the branch point. OSC uses candidate
rank, candidate count, and a coefficient retained from an earlier batch;
corrected values are propagated into token advantages before the policy update.

## Configuration changes

All keys below are relative to `actor_rollout_ref.rollout`:

- `expansion_score_norm=none` selects the unstandardized scoring path.
- `postsel_correction_weight=0` disables OSC.
- `num_branches=1` changes the number of fresh branches per selected slot.
  To retain the nominal 22-leaf budget with the default initial rollout count
  and two rounds, use `beam_size=6` as well.
- `expansion_mode` selects the candidate criterion supported by
  `sample_expansion_nodes`.

The configuration switches expose mechanisms; changing one may also change
sampling behavior. Keep rollout count, training settings, and evaluation
inputs consistent when setting up comparisons.

## Diagnostics

`tree_stats_dump.py` records selection metadata, branch structure, and leaf
statistics when `TREE_STATS_PATH` is set. For example:

```bash
export TREE_STATS_PATH=runs/diagnostics/tree_stats.jsonl
```

Launch training from the repository root after setting it. The diagnostic
helper uses per-process files for concurrent workers. Diagnostic outputs are
excluded from version control by default.

Its fallback step field counts dump calls within a worker process. Validation
calls can also advance this counter; disable periodic validation with
`--override trainer.test_freq=-1` when collecting traces that need a direct
one-to-one mapping from this counter to training steps. A resumed process starts
its counter again unless a label is provided explicitly.

The local `metrics.jsonl` file stores one JSON object per logging event with a
`step` field. `--dump-rollouts` enables additional trajectory output through
the existing trainer. None of these generated files are part of the source
release.

## Release adaptations

The SIPO rollout, tree credit computation, actor losses, reward function, and
trainer loop retain their source computation. Release-specific changes provide
portable input/output paths, a local retrieval connector, local logging, and
CLI-driven retrieval startup. The retrieval connector preserves the original
query payload and passage formatting and applies the configured request timeout.
Upstream alternate backend code is retained; the installation instructions focus
on the SIPO FSDP/vLLM path.
