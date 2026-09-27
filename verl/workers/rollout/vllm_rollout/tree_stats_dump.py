"""Zero-side-effect tree instrumentation for the AT2PO offline tree rollout worker.

Two output files, both gated on the env var TREE_STATS_PATH (unset -> every function
returns immediately, so this module can live in the tree permanently):

  $TREE_STATS_PATH                    one JSONL record per tree node per step.
                                      Schema is a superset of the original, so
                                      rho_estimator.py reads it unchanged.
  $TREE_STATS_PATH + ".expansion"     one JSONL record per expansion CANDIDATE per
                                      iteration (rec_type="selection"), and one per
                                      created sibling (rec_type="branch").

The second file is what makes the selection-bias measurement possible: it pairs each
selected incumbent with the sibling that was forked next to it, and it preserves the
full candidate set with the scores that produced the ranking, so alternative selection
criteria can be scored counterfactually offline with no extra rollouts.

Read-only with respect to training. It reads node attributes, sets a few private
attributes that nothing else in the worker reads (`_raw_score`, `_sel_iters`,
`_created_iter`, `_forked_from`, `_branch_idx`, `_sibling_added_iter`), and appends to
files.

Usage -- see APPLY_PATCH.md for the insertion points.
"""

import json
import os
import threading

_LOCK = threading.Lock()
_CALLS = [0]          # dump_tree_stats invocations in this process, used as a step fallback


def _path():
    return os.environ.get("TREE_STATS_PATH", "")


def _step():
    """Training step label.

    TREE_STATS_STEP is static for a whole job, so relying on it would stamp every
    record with the same value -- which not only kills the per-step drift measurement
    but risks tree_uid collisions when records from different steps are keyed together.
    Fall back to a per-process counter of completed dump_tree_stats calls, i.e. of
    completed training steps. All 8 worker processes advance in lockstep so their
    counters agree. (Assumes no validation passes interleave: validation also reaches
    dump_tree_stats, so run diagnostics with TEST_FREQ=-1.)
    """
    try:
        s = int(os.environ.get("TREE_STATS_STEP", "-1"))
    except ValueError:
        s = -1
    return s if s >= 0 else _CALLS[0]


def _append(path, recs):
    """Append records to a PER-PROCESS file.

    The rollout worker runs as 8 separate Ray actors (dp=8, TP=1), all of which reach
    this code in the same training step, and the share is NFS. A threading.Lock does not
    serialise across processes, and a large buffered append is not atomic on NFS, so a
    single shared file would interleave and corrupt lines. One file per pid removes the
    contention entirely; the analysis globs them back together.
    """
    if not recs:
        return
    path = f"{path}.p{os.getpid()}"
    blob = "".join(json.dumps(r) + "\n" for r in recs)
    with _LOCK:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as f:
            f.write(blob)


def stash_raw_leaf_scores(root_nodes):
    """Call BEFORE the `if self.leaf_value_norm:` block, which overwrites leaf.value."""
    if not _path():
        return
    for root in root_nodes:
        for leaf in root.get_all_leaves():
            leaf._raw_score = leaf.value


def _num_existing_branches(node):
    """MIRROR of the live selection penalty term.

    Source of truth: vllm_rollout_with_tools_tree_offline.py L379-382. Duplicated
    rather than imported so that instrumentation cannot alter selection behaviour.
    If that block ever changes, this goes stale -- the recorded `score` is the only
    thing affected, and it can be re-derived from the other recorded fields.
    """
    if node.parent_node is None:
        return max(0, len(node.child_nodes) - 1)
    return max(0, len(node.parent_node.child_nodes) - 1)


def _live_score(node):
    """MIRROR of the live selection score: L375 (prob = entropy_now) + L386 (additive
    penalty). Note `entropy` here is path-average surprisal, not a distributional
    entropy -- see AT2PO_CODE_FINDINGS.md F3."""
    ent = getattr(node, "entropy", None)
    if ent is None:
        return None
    return float(ent) - 0.05 * _num_existing_branches(node)


def record_selection(root, exp_iter, selected_nodes):
    """Call immediately AFTER sample_expansion_nodes() returns, per tree, BEFORE any
    branch is created -- the candidate set must be recorded in the state that produced
    the ranking.

    Records every candidate, not only the winners, so that the realised deterministic
    top-k can be compared offline against alternative criteria on the same candidates.
    """
    path = _path()
    if not path:
        return
    step = _step()
    sel_uids = {n.node_uid for n in (selected_nodes or [])}
    recs = []
    for n in root.get_non_leaf_nodes():
        chosen = n.node_uid in sel_uids
        if chosen:
            if not hasattr(n, "_sel_iters"):
                n._sel_iters = []
            n._sel_iters.append(exp_iter)
        recs.append({
            "rec_type": "selection",
            "step": step,
            "tree_uid": root.tree_uid,
            "exp_iter": exp_iter,
            "node_uid": n.node_uid,
            "parent_uid": n.parent_node.node_uid if n.parent_node is not None else None,
            "depth": n.depth,
            "entropy": _f(getattr(n, "entropy", None)),
            "initial_entropy": _f(getattr(n, "initial_entropy", None)),
            "n_children_at_time": len(n.child_nodes),
            "num_existing_branches": _num_existing_branches(n),
            "score": _f(_live_score(n)),
            "selected": chosen,
            # leaves under this node right now. With num_branches=1 an unexpanded
            # candidate has exactly one, which is the incumbent leaf for pairing.
            "leaves_at_time": [l.node_uid for l in n.get_all_leaves()],
        })
    _append(path + ".expansion", recs)


def mark_new_branch(new_branch, incumbent, exp_iter, branch_idx=0):
    """Call right after `new_branches.append(new_branch)`.

    `incumbent` is the SELECTED node (the worker's `node`); the new branch is created
    from `incumbent.parent_node`, so the two are siblings. That asymmetry -- incumbent
    chosen by top-k on its own realised surprisal, sibling drawn fresh -- is the whole
    point of the measurement.

    `branch_idx` distinguishes multiple fresh siblings created from the same incumbent
    when `num_branches > 1`. Two fresh siblings of the same incumbent ARE exchangeable
    with each other, so the contrast between them is the unbiased comparison, while
    incumbent-vs-fresh is the biased one. Recording the index lets both be computed from
    a single run, as a within-run control. It defaults, so callers that predate the
    `num_branches` knob keep working unchanged.
    """
    path = _path()
    if not path:
        return
    new_branch._created_iter = exp_iter
    new_branch._forked_from = incumbent.node_uid
    new_branch._branch_idx = branch_idx
    incumbent._sibling_added_iter = exp_iter
    _append(path + ".expansion", [{
        "rec_type": "branch",
        "step": _step(),
        "tree_uid": getattr(incumbent, "tree_uid", None),
        "exp_iter": exp_iter,
        "branch_idx": branch_idx,
        "new_branch_uid": new_branch.node_uid,
        "incumbent_uid": incumbent.node_uid,
        "parent_uid": new_branch.parent_node.node_uid if new_branch.parent_node is not None else None,
        "incumbent_depth": incumbent.depth,
        "incumbent_entropy": _f(getattr(incumbent, "entropy", None)),
        # Confound guard: the worker sets new_branch.call_counter = max(0, node.call_counter - 1)
        # (L1445). The max(0, ...) clamp means that when the incumbent had call_counter == 0
        # the sibling starts with the SAME budget rather than one fewer, i.e. it gets an extra
        # tool call relative to the incumbent's realised path. Record both so those pairs can
        # be excluded from the F4 estimate instead of silently biasing it.
        "incumbent_call_counter": getattr(incumbent, "call_counter", None),
        "new_branch_call_counter": getattr(new_branch, "call_counter", None),
        "incumbent_leaves": [l.node_uid for l in incumbent.get_all_leaves()],
    }])


def _leaf_count(node, cache):
    uid = node.node_uid
    if uid not in cache:
        cache[uid] = 1 if node.is_leaf else len(node.get_all_leaves())
    return cache[uid]


def dump_tree_stats(root_nodes, out_path, step, node_value_mode=None, node_adv_mode=None):
    """Call AFTER the node_adv_mode dispatch, BEFORE step 4 (token-level assignment)."""
    if not out_path:
        return
    if step is None or step < 0:
        step = _CALLS[0]
    recs = []
    for root in root_nodes:
        cache = {}
        nodes = [root] + root.get_subtree_nodes()
        for n in nodes:
            plen = 0
            if n.parent_node is not None:
                plen = len(n.parent_node.curr_token_ids) - len(n.parent_node.prompt_token_ids)
            nlen = len(n.curr_token_ids) - len(n.prompt_token_ids)
            recs.append({
                "step": step,
                "tree_uid": root.tree_uid,
                "node_uid": n.node_uid,
                "parent_uid": n.parent_node.node_uid if n.parent_node is not None else None,
                "depth": n.depth,
                "is_leaf": bool(n.is_leaf),
                "n_children": len(n.child_nodes),
                "n_leaves": _leaf_count(n, cache),
                # incremental tokens: the span this node's advantage is written into
                "tok_span": max(0, nlen - plen),
                "entropy": _f(getattr(n, "entropy", None)),
                "initial_entropy": _f(getattr(n, "initial_entropy", None)),
                "value": _f(getattr(n, "value", None)),
                "advantage": _f(getattr(n, "advantage", None)),
                "raw_score": _f(getattr(n, "_raw_score", None)),
                # expansion provenance (None on nodes that predate any expansion)
                "created_iter": getattr(n, "_created_iter", None),
                "forked_from": getattr(n, "_forked_from", None),
                "branch_idx": getattr(n, "_branch_idx", None),
                "sibling_added_iter": getattr(n, "_sibling_added_iter", None),
                "sel_iters": list(getattr(n, "_sel_iters", []) or []),
                "node_value_mode": node_value_mode,
                "node_adv_mode": node_adv_mode,
            })
    _append(out_path, recs)
    _CALLS[0] += 1          # advance the step label for the next rollout's hooks


def _f(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None
