# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
import importlib
import logging
import os
import time
import random
from copy import deepcopy
from typing import Dict, List, Counter, Optional
from enum import Enum

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length, masked_mean
from verl.workers.rollout.tools.base_tool import BaseTool
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import (
    vLLMRollout,
    _pre_process_inputs,
    _repeat_interleave,
)
from verl.trainer.ppo import core_algos
from verl.workers.rollout.vllm_rollout.tree_stats_dump import (
    stash_raw_leaf_scores, dump_tree_stats, record_selection, mark_new_branch)

# ---------------------------------------------------------------------------
# State and helpers for the post-selection (order-statistic) correction.
#
# `_A2_STATE` carries the correlation between the expansion criterion and the
# outcome from one training step to the next, so the correction only ever uses
# information that was available BEFORE the batch it corrects. One dict per
# worker process; the 8 rollout actors advance in lockstep so their estimates
# agree to sampling noise.
_A2_STATE = {"a2": None, "n": 0}
_EZ_CACHE = {}


def _expected_order_stat(r, n):
    """E[Z_(r:n)], the r-th LARGEST of n standard normals, via Blom's approximation.

    Verified against Monte Carlo (40k reps) at the ranks and candidate counts this
    configuration produces: max absolute error 0.008, e.g. rank 1 of 22 gives 1.9096
    against 1.9078, rank 6 of 22 gives 0.6657 against 0.6675.
    """
    if n is None or r is None or n < 1 or r < 1 or r > n:
        return 0.0
    key = (int(r), int(n))
    if key not in _EZ_CACHE:
        from statistics import NormalDist
        _EZ_CACHE[key] = NormalDist().inv_cdf((n - r + 1 - 0.375) / (n + 0.25))
    return _EZ_CACHE[key]


def _candidate_level_a2(root_nodes):
    """corr(expansion criterion, outcome) over ALL candidates, not just the winners.

    Measured on real dumps, the same correlation computed on SELECTED units only is
    inflated 1.1-2.5x by range restriction, and the inflation depends on which statistic
    the selection used -- so a selected-only estimate would systematically over-correct.
    Every non-leaf node is a candidate and has a subtree whose leaves carry values, so the
    unrestricted estimate is available here at no extra cost.
    """
    xs, ys = [], []
    for root in root_nodes:
        for node in root.get_non_leaf_nodes():
            e = getattr(node, "entropy", None)
            if e is None:
                continue
            vs = [l.value for l in node.get_all_leaves() if l.value is not None]
            if not vs:
                continue
            xs.append(float(e))
            ys.append(sum(vs) / len(vs))
    n = len(xs)
    if n < 50:
        return None, n
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx < 1e-12 or sy < 1e-12:
        return None, n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / (sx * sy), n
# ---------------------------------------------------------------------------
from verl.trainer.ppo.reward import compute_reward, load_reward_manager
import math
import uuid

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator"""
    GAE = "gae"
    GRPO = "grpo"
    GRPO_PASSK = "grpo_passk"


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


def _load_tool_from_config(tool_config: DictConfig) -> BaseTool:
    """Dynamically loads a tool from its configuration."""
    module_path, class_name = tool_config.class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)

        tool_class = getattr(module, class_name)

        tool_params = OmegaConf.to_container(
            tool_config.get("params", {}), resolve=True
        )

        tool_instance = tool_class(**tool_params)

        return tool_instance
    except ImportError as e:
        logger.error(f"Failed to import module {module_path}: {e}")
        raise
    except AttributeError as e:
        logger.error(f"Failed to find class {class_name} in module {module_path}: {e}")
        raise
    except TypeError as e:
        logger.error(
            f"Failed to instantiate {class_name} with provided parameters: {e}"
        )
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error loading tool from {tool_config.class_path}: {e}"
        )
        raise

class ToolTreeNode:
    """Tree node for managing tool-based generation with tree search."""
    
    def __init__(
        self,
        tree_uid: str,
        node_uid: str,
        prompt_token_ids: List[int],
        curr_token_ids: List[int],
        result_mask: List[int],
        call_counter: int = 0,
        parent_node: Optional['ToolTreeNode'] = None,
        is_root: bool = False,
        is_active: bool = True,
        is_leaf: bool = False,
        depth: int = 0,
        entropy: float = 0.0,
    ):
        self.tree_uid = tree_uid
        self.node_uid = node_uid
        
        # Token sequences
        self.prompt_token_ids = prompt_token_ids
        self.curr_token_ids = curr_token_ids.copy()
        self.result_mask = result_mask.copy()
        
        # Tool call tracking
        self.call_counter = call_counter
        
        # Tree structure
        self.parent_node = parent_node
        self._child_nodes = []
        
        # Node status
        self.is_root = is_root
        self.is_active = is_active
        self.is_leaf = is_leaf
        self.depth = depth
        
        # Entropy tracking
        self.entropy = entropy
        self.initial_entropy = entropy if is_root else 0.0
        
        # Rollout tracking
        self.last_rollout_length = 0  # Length of the last rollout (generated tokens)
        
        # Value tracking (reward from reward model)
        self.value = None  # Will be set after reward computation
        self.advantage = None
    
    
    def __str__(self):
        """Return a detailed string representation of the node."""
        parent_uid = self.parent_node.node_uid if self.parent_node else "None"
        num_children = len(self._child_nodes)
        response_len = len(self.curr_token_ids) - len(self.prompt_token_ids)
        value_str = f"{self.value:.4f}" if self.value is not None else "None"
        advantage_str = f"{self.advantage:.4f}" if hasattr(self, 'advantage') and self.advantage is not None else "None"
        
        return (
            f"ToolTreeNode(uid={self.node_uid}, "
            f"tree={self.tree_uid[:8]}..., "
            f"parent={parent_uid}, "
            f"children={num_children}, "
            f"depth={self.depth}, "
            f"is_root={self.is_root}, "
            f"is_leaf={self.is_leaf}, "
            f"is_active={self.is_active}, "
            f"call_counter={self.call_counter}, "
            f"response_len={response_len}, "
            f"last_rollout_length={self.last_rollout_length}, "
            f"entropy={self.entropy:.4f}, "
            f"init_entropy={self.initial_entropy:.4f}, "
            f"value={value_str}, "
            f"advantage={advantage_str})"
        )
        
    def add_child(self, child_node: 'ToolTreeNode'):
        """Add a child node."""
        if child_node is self or child_node.node_uid == self.node_uid:
            raise ValueError("A node cannot be its own child")
        self._child_nodes.append(child_node)
    
    @property
    def child_nodes(self):
        return self._child_nodes
    
    def get_subtree_nodes(self):
        """Get all descendant nodes."""
        nodes = []
        nodes_to_visit = list(self._child_nodes)
        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            nodes.append(current_node)
            nodes_to_visit.extend(current_node._child_nodes)
        return nodes
    
    def get_all_leaves(self):
        """Get all leaf nodes."""
        leaves = []
        if self.is_leaf:
            leaves.append(self)
        for node in self.get_subtree_nodes():
            if node.is_leaf:
                leaves.append(node)
        return leaves
    
    def get_non_leaf_nodes(self):
        """Get all non-leaf nodes that can be expanded."""
        non_leaves = []
        if not self.is_leaf:
            non_leaves.append(self)
        for node in self.get_subtree_nodes():
            if not node.is_leaf:
                non_leaves.append(node)
        return non_leaves
    
    def sample_expansion_nodes(self, n: int, mode: str = 'random',
                              entropy_weight: float = 0.0,
                              branch_probability: float = 0.5,
                              score_norm: str = 'none',
                              penalty_lambda: float = 0.05):
        """Sample n nodes for expansion.
        
        Args:
            n: Number of nodes to sample
            mode: Selection mode, either 'random' or 'entropy'
                - 'random': randomly select nodes
                - 'entropy': select nodes with highest probability scores
            entropy_weight: Weight for entropy delta in probability calculation
            branch_probability: Base branch probability threshold (not used in offline mode)
        """
        candidate_nodes = self.get_non_leaf_nodes()
        if not candidate_nodes:
            return []
        
        if mode == 'random':
            if len(candidate_nodes) >= n:
                return random.sample(candidate_nodes, n)
            else:
                return random.choices(candidate_nodes, k=n)
        
        elif mode == 'entropy':
            # Calculate probability score for each node
            node_scores = []
            raw = []
            pen = []
            for node in candidate_nodes:
                entropy_now = node.entropy
                entropy_init = node.initial_entropy
                entropy_delta = entropy_now - entropy_init

                # Calculate base probability with random component and entropy delta
                # prob = random.random() + entropy_weight * entropy_delta
                # prob = entropy_weight * entropy_delta
                prob = entropy_now

                # Apply node-level branch penalty based on existing children
                # If a node has already been expanded (has children), penalize further expansion
                if node.parent_node is None:
                    num_existing_branches = max(0, len(node.child_nodes) - 1)
                else:
                    num_existing_branches = max(0, len(node.parent_node.child_nodes) - 1)
                penalty_factor = 1.0 - 0.05 * num_existing_branches
                # penalty_factor = max(0.0, penalty_factor)  # Ensure not negative
                # prob = prob * penalty_factor
                raw.append(float(prob) if prob is not None else 0.0)
                pen.append(float(num_existing_branches))
                prob = prob-0.05*num_existing_branches


                node_scores.append((node, prob))

            # Optional: put the criterion and the penalty on the same scale before
            # combining them.
            #
            # The shipped rule adds an ABSOLUTE constant (0.05 per existing sibling) to a
            # criterion measured in nats, which has two measured consequences:
            #   * the root accumulates `initial_rollouts` children, so every candidate at
            #     the pre-first-tool-call depth carries a flat ~0.5 handicap purely from
            #     its parent's fan-out -- a stratum-level bias nobody chose, which changes
            #     60-74% of expansion decisions;
            #   * path-average surprisal shrinks as the policy sharpens, so the fixed
            #     constant's relative weight grows (4.2% -> 9.7% of the signal over 8
            #     steps), silently drifting the rule from criterion-ranked toward
            #     breadth-first over training.
            # Standardising the criterion within the candidate set makes the two terms
            # commensurable and removes both effects at once. `none` reproduces the
            # shipped behaviour exactly.
            if score_norm == "zscore" and len(raw) >= 2:
                m = sum(raw) / len(raw)
                sd = (sum((x - m) ** 2 for x in raw) / len(raw)) ** 0.5
                if sd > 1e-9:
                    node_scores = [(nd, (r - m) / sd - penalty_lambda * p)
                                   for (nd, _), r, p in zip(node_scores, raw, pen)]
            # print('node probs:', [node_score[1] for node_score in node_scores])
            # Sort nodes by probability score (high to low)
            sorted_nodes = sorted(node_scores, key=lambda x: x[1], reverse=True)
            
            # Tag the winners with their rank and the candidate count. Only used by the
            # post-selection correction (see Step 1c); harmless otherwise.
            for _i, (_nd, _sc) in enumerate(sorted_nodes):
                _nd._sel_rank = _i + 1
                _nd._sel_ncand = len(sorted_nodes)

            # Select top n nodes
            if len(sorted_nodes) >= n:
                return [node for node, score in sorted_nodes[:n]]
            else:
                # If not enough nodes, duplicate the highest probability nodes
                result = [node for node, score in sorted_nodes]
                while len(result) < n:
                    additional = [node for node, score in sorted_nodes[:min(n - len(result), len(sorted_nodes))]]
                    result.extend(additional)
                return result[:n]
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'random' or 'entropy'")
    
    def sample_leaves(self, n: int):
        """Sample n leaf nodes with pruning and duplication.
        
        If there are fewer leaves than needed, duplicate the last leaf as new nodes
        in the tree (siblings of the last leaf).
        If there are more leaves than needed, unsampled leaves and their branches will be pruned.
        
        Args:
            n: Number of leaf nodes to sample
            
        Returns:
            List of n sampled leaf nodes
        """
        all_leaves = self.get_all_leaves()
        
        if len(all_leaves) < n:
            # Duplicate the last leaf if needed - create new leaf nodes in the tree
            logger.warning(f"Not enough leaves ({len(all_leaves)}) for sampling {n}, will duplicate last leaf")
            result = all_leaves.copy()
            
            if not all_leaves:
                logger.error("No leaves available to duplicate")
                return result
            
            # Only duplicate the last leaf
            last_leaf = all_leaves[-1]
            parent_node = last_leaf.parent_node
            
            if parent_node is None:
                # If last leaf is root (edge case), just return what we have
                logger.warning(f"Last leaf {last_leaf.node_uid} is root, cannot create sibling")
                return result
            
            # Create new leaf nodes as siblings of the last leaf
            duplicate_count = 0
            while len(result) < n:
                duplicate_count += 1
                new_leaf_uid = f"{last_leaf.node_uid}_dup{duplicate_count}"
                new_leaf = ToolTreeNode(
                    tree_uid=last_leaf.tree_uid,
                    node_uid=new_leaf_uid,
                    prompt_token_ids=last_leaf.prompt_token_ids,
                    curr_token_ids=last_leaf.curr_token_ids.copy(),
                    result_mask=last_leaf.result_mask.copy(),
                    call_counter=last_leaf.call_counter,
                    parent_node=parent_node,
                    is_root=False,
                    is_active=False,
                    is_leaf=True,
                    depth=last_leaf.depth,
                    entropy=last_leaf.entropy,
                )
                new_leaf.initial_entropy = last_leaf.initial_entropy
                new_leaf.last_rollout_length = last_leaf.last_rollout_length
                
                # Add new leaf to parent's children
                parent_node.add_child(new_leaf)
                result.append(new_leaf)
            
            logger.info(f"Created {duplicate_count} duplicate leaf nodes from last leaf in tree")
            return self.get_all_leaves()
        
        elif len(all_leaves) > n:
            # Prune unsampled leaves and their branches
            sampled_leaves = random.sample(all_leaves, n)
            unsampled_leaves = [leaf for leaf in all_leaves if leaf not in sampled_leaves]
            
            logger.info(f"Pruning {len(unsampled_leaves)} unsampled leaves and their branches")
            
            # Prune each unsampled leaf's branch
            for leaf in unsampled_leaves:
                self._prune_branch(leaf)
            
            return self.get_all_leaves()
        
        else:
            # Exactly n leaves, return all
            return all_leaves
    
    def _prune_branch(self, leaf_node: 'ToolTreeNode'):
        """Prune a branch by removing a leaf node and recursively pruning upward.
        
        This method removes the leaf node from its parent's children list.
        If the parent becomes childless after removal (no other children remain),
        it recursively prunes the parent as well, continuing upward until reaching
        a node that has other children (paths to other retained leaves) or the root.
        
        Args:
            leaf_node: The leaf node to prune
        """
        if not leaf_node.is_leaf:
            logger.warning(f"Attempting to prune non-leaf node {leaf_node.node_uid}")
            return
        
        current_node = leaf_node
        
        # Recursively prune upward
        while current_node is not None:
            parent = current_node.parent_node
            
            if parent is None:
                # Reached root, cannot prune further
                logger.warning(f"Cannot prune root node {current_node.node_uid}")
                break
            
            # Remove current node from parent's children
            if current_node in parent._child_nodes:
                parent._child_nodes.remove(current_node)
                logger.debug(f"Pruned node {current_node.node_uid} from parent {parent.node_uid}")
            else:
                logger.warning(f"Node {current_node.node_uid} not found in parent's children")
                break
            
            # Check if parent still has other children
            if len(parent._child_nodes) > 0:
                # Parent has other children (paths to other retained leaves), stop pruning
                logger.debug(f"Parent {parent.node_uid} still has {len(parent._child_nodes)} children, stopping pruning")
                break
            
            # Parent has no children left, continue pruning upward
            logger.debug(f"Parent {parent.node_uid} has no children left, continuing to prune upward")
            current_node = parent
    
    def create_branch(self, branch_id: int = 0):
        """Create a new branch from this node."""
        child_uid = f"{self.node_uid}_b{branch_id}"
        child_node = ToolTreeNode(
            tree_uid=self.tree_uid,
            node_uid=child_uid,
            prompt_token_ids=self.prompt_token_ids,
            curr_token_ids=self.curr_token_ids.copy(),
            result_mask=self.result_mask.copy(),
            call_counter=self.call_counter,
            parent_node=self,
            is_root=False,
            is_active=True,
            is_leaf=False,
            depth=self.depth,
            entropy=self.entropy,
        )
        # Inherit initial_entropy from root or parent
        if self.is_root:
            # If this is root, child should use root's entropy as initial
            child_node.initial_entropy = self.entropy if self.entropy > 0.0 else 0.0
        else:
            # If this is not root, inherit from parent's initial_entropy
            child_node.initial_entropy = self.initial_entropy
        
        self.add_child(child_node)
        return child_node
    
    def compute_value_from_children(self, mode: str = 'child_mean'):
        """Recursively compute node value based on child values.
        
        Args:
            mode: Computation mode
                - 'child_mean': Node value is the mean of all child node values
                - 'child_softmax': Node value is the weighted sum of child values,
                                   weighted by softmax of child entropies
        
        Returns:
            The computed value for this node
        """
        if self.is_leaf:
            # Leaf node value already set from reward model
            return self.value
        
        # Non-leaf node: compute value from children
        child_values = []
        child_entropies = []
        for child in self.child_nodes:
            child_value = child.compute_value_from_children(mode)
            if child_value is not None:
                child_values.append(child_value)
                child_entropies.append(child.entropy)
        
        if not child_values:
            # No valid child values, set to 0
            self.value = 0.0
            logger.warning(f"Node {self.node_uid}: no valid child values, set value=0.0")
            return self.value
        
        # Compute value based on mode
        if mode == 'child_mean':
            self.value = np.mean(child_values)
        elif mode == 'child_softmax':
            # Compute softmax weights based on child entropies
            child_entropies_array = np.array(child_entropies)
            # Apply softmax: exp(entropy) / sum(exp(entropy))
            exp_entropies = np.exp(child_entropies_array)
            softmax_weights = exp_entropies / np.sum(exp_entropies)
            
            # Weighted sum of child values
            child_values_array = np.array(child_values)
            self.value = np.sum(softmax_weights * child_values_array)
            
            logger.debug(f"Node {self.node_uid}: child_softmax weights={softmax_weights}, value={self.value:.4f}")
        else:
            raise ValueError(f"Unsupported mode: {mode}. Must be 'child_mean' or 'child_softmax'")
        
        return self.value
    
class vLLMRolloutWithTools(vLLMRollout):
    """
    An advanced vLLM rollout engine capable of handling multiple tools like
    code interpreters and search engines during generation.

    This class extends vLLMRollout by orchestrating a multi-step generation
    process where the language model can emit special tokens to trigger external
    tools. The tool outputs are then fed back into the model to continue
    generation.
    """

    def __init__(
        self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs
    ):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer

        # initial_rollouts: number of initial child nodes to create from each root
        # used when enable_dynamic_rollouts is **False**
        self.initial_rollouts = self.config.initial_rollouts
        # Get beam search related parameters from config
        self.beam_size = self.config.beam_size
        # Number of NEW siblings created per selected node. Historically hardcoded to 1,
        # which makes each branch point a pair of (selected incumbent, one fresh draw) --
        # not exchangeable, since the incumbent was chosen by top-k on its own realised
        # segment. With num_branches >= 2 a branch point yields >=2 fresh draws that ARE
        # exchangeable with each other. Default 1 reproduces the previous behaviour exactly.
        #
        # Read defensively: on a struct-mode DictConfig a missing key raises, and the exact
        # exception type is an omegaconf detail, so do not rely on getattr's AttributeError
        # contract here. A config without this key must keep working unchanged.
        try:
            self.num_branches = max(1, int(self.config.num_branches))
        except Exception:
            self.num_branches = 1
        # Which statistic `node.entropy` holds, hence what the expansion rule ranks by.
        #   "surprisal" (default, historical): mean -log p of the SAMPLED tokens.
        #   "plugin": mean top-k plug-in entropy of the predictive distribution, i.e.
        #             the quantity "Entropy-Guided Tree Expansion" names.
        # Default preserves existing behaviour byte for byte.
        try:
            self.entropy_estimator = str(self.config.entropy_estimator)
        except Exception:
            self.entropy_estimator = "surprisal"
        if self.entropy_estimator not in ("surprisal", "plugin"):
            raise ValueError(f"entropy_estimator must be 'surprisal' or 'plugin', "
                             f"got {self.entropy_estimator!r}")
        # Weight of the within-outcome-group cost tie-break (see Step 1b). 0 disables it
        # and reproduces the previous behaviour exactly.
        try:
            self.cost_tiebreak_weight = float(self.config.cost_tiebreak_weight)
        except Exception:
            self.cost_tiebreak_weight = 0.0
        # How the expansion criterion and its sibling penalty are combined.
        #   "none"   (default) shipped behaviour: absolute nats minus 0.05*siblings
        #   "zscore" standardise the criterion within the candidate set first, so the
        #            two terms are commensurable -- removes both the fan-out-induced
        #            stratum handicap and the drift of the penalty's relative weight.
        try:
            self.expansion_score_norm = str(self.config.expansion_score_norm)
        except Exception:
            self.expansion_score_norm = "none"
        if self.expansion_score_norm not in ("none", "zscore"):
            raise ValueError(f"expansion_score_norm must be 'none' or 'zscore', "
                             f"got {self.expansion_score_norm!r}")
        try:
            self.expansion_penalty_lambda = float(self.config.expansion_penalty_lambda)
        except Exception:
            self.expansion_penalty_lambda = 0.05
        # The sibling penalty counts EXISTING SIBLINGS, not expansion events, so with
        # num_branches = B every expansion adds B siblings and the penalty per expansion is
        # B*lambda rather than lambda. lambda was implicitly set under B = 1, so raising B
        # silently doubles the penalty's strength.
        #
        # This shows up in the ablation on Qwen2.5-7B multihop: score normalisation alone is
        # +1.73 and exchangeable branching alone is +1.46, but the two together are +0.40.
        # The only difference between the exchangeable arm and the combined arm is the
        # normalisation, so the interaction is the normalised criterion meeting a penalty
        # that is twice as strong as it was tuned for.
        #
        # Enabling this divides lambda by B, restoring a constant penalty per expansion
        # event. Default False reproduces the shipped behaviour exactly.
        #
        # Side effect worth knowing: the root already holds `initial_rollouts` children
        # before any expansion, so its children carry a handicap of lambda*(M-1) that has
        # nothing to do with B. Dividing lambda also shrinks that handicap by B. That
        # handicap is itself an artefact we argue against, so the direction is right, but it
        # means this flag changes two things rather than one.
        try:
            self.expansion_penalty_per_branch = bool(self.config.expansion_penalty_per_branch)
        except Exception:
            self.expansion_penalty_per_branch = False
        # Weight of the post-selection correction (Step 1c). 0 disables it entirely and
        # reproduces the previous behaviour; 1.0 applies the full order-statistic estimate.
        try:
            self.postsel_correction_weight = float(self.config.postsel_correction_weight)
        except Exception:
            self.postsel_correction_weight = 0.0
        self.samples_per_tree = self.config.samples_per_tree
        self.expansion_mode = self.config.expansion_mode
        self.expansion_iterations = self.config.expansion_iterations
        self.branch_probability = self.config.branch_probability
        self.entropy_weight = self.config.entropy_weight
        # Total outputs per input sample = samples_per_tree (samples from the tree)
        # Each input has 1 root with initial_rollouts initial branches, then expanded, then sampled
        assert self.config.n == self.samples_per_tree

        # Validate expansion_mode
        if self.expansion_mode not in ['random', 'entropy']:
            raise ValueError(f"Invalid expansion_mode: {self.expansion_mode}. Must be 'random' or 'entropy'")
        
        # Validate samples_per_tree
        if self.samples_per_tree < 1:
            raise ValueError(f"samples_per_tree must be >= 1, got {self.samples_per_tree}")

        self.enable_dynamic_rollouts = self.config.enable_dynamic_rollouts
        logger.info(f"enable_dynamic_rollouts: {self.enable_dynamic_rollouts}")
        logger.info(f"beam_size: {self.beam_size}, samples_per_tree: {self.samples_per_tree}")
        logger.info(f"expansion_mode: {self.expansion_mode}, expansion_iterations: {self.expansion_iterations}")
        
        # Reward and advantage computation
        self.reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        self.use_kl_in_reward = self.config.use_kl_in_reward
        self.kl_penalty = self.config.kl_penalty
        self.adv_estimator = self.config.adv_estimator
        self.gamma = self.config.gamma
        self.lam = self.config.lam
        self.norm_adv_by_std_in_grpo = self.config.norm_adv_by_std_in_grpo
        ##
        self.leaf_value_norm = self.config.leaf_value_norm
        self.node_value_mode = self.config.node_value_mode  # child_mean/leaf_mean/child_softmax
        self.node_adv_mode = self.config.node_adv_mode  #vanilla/node_value/diff_parent/...
        
        # Initialize KL controller if needed
        if self.use_kl_in_reward:
            kl_ctrl_config = self.config.kl_ctrl
            self.kl_ctrl = core_algos.AdaptiveKLController(
                init_kl_coef=kl_ctrl_config.get('init_kl_coef', 0.1),
                target_kl=kl_ctrl_config.get('target_kl', 6.0),
                horizon=kl_ctrl_config.get('horizon', 10000)
            )
        else:
            self.kl_ctrl = None


        # Get tool settings from config
        tools_config = self.config.get("tools", OmegaConf.create({}))

        # Get general tool configuration
        self.tool_call_limit = tools_config.get("call_limit", 6)
        self.max_tool_workers = tools_config.get("max_workers", 64)
        self.tool_timeout = tools_config.get("timeout", 120)

        # Other possible general tool configurations
        self.tool_retry_count = tools_config.get("retry_count", 4)
        self.tool_verbose_logging = tools_config.get("verbose_logging", False)

        self.tools: Dict[str, BaseTool] = {}
        if "tool_instances" in tools_config:
            for tool_name, tool_config in tools_config.tool_instances.items():
                logger.info(f"Loading tool '{tool_name}' from {tool_config.class_path}")
                try:
                    tool_instance = _load_tool_from_config(tool_config)
                    self.tools[tool_instance.trigger_tag] = tool_instance
                except Exception as e:
                    logger.error(
                        f"Could not initialize tool '{tool_name}'. Please check your configuration. Error: {e}"
                    )
                    if tools_config.get("fail_on_error", False):
                        raise

        self.stop_sequences = [f"</{tag}>" for tag in self.tools.keys()]
        self.logprobs = 10  # entropy
        self.initial_entropy_dict = {}  # Record initial entropy for each active index in first round

        if not self.tools:
            logger.warning(
                "vLLMRolloutWithTools initialized, but no tools were configured."
            )

        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_tool_workers
        )

    def __del__(self):
        self.executor.shutdown(wait=False)


    def _build_initial_trees(self, prompt_token_ids_list: List[List[int]], 
                            initial_rollouts_list: List[int]) -> List[ToolTreeNode]:
        """Build initial tree structures with root nodes.
        
        Args:
            prompt_token_ids_list: List of prompt token IDs for each input sample
            initial_rollouts_list: Number of initial child nodes to create from each root
            
        Returns:
            List of root nodes (one per input sample), each with initial_rollouts child nodes
        """
        root_nodes = []
        
        for i, prompt_ids in enumerate(prompt_token_ids_list):
            initial_rollouts = initial_rollouts_list[i]
            
            # Create one root node for this input sample
            tree_uid = str(uuid.uuid4())
            node_uid = f"sample_{i}_root"
            
            root_node = ToolTreeNode(
                tree_uid=tree_uid,
                node_uid=node_uid,
                prompt_token_ids=prompt_ids,
                curr_token_ids=prompt_ids.copy(),
                result_mask=[],
                call_counter=0,
                parent_node=None,
                is_root=True,
                is_active=True,
                is_leaf=False,
                depth=0,
                entropy=0.0,
            )
            
            # Create initial_rollouts child nodes from this root as the first rollout
            for branch_idx in range(initial_rollouts):
                child_node = root_node.create_branch(branch_idx)
            
            root_nodes.append(root_node)
        
        return root_nodes




    def _extract_content(self, text: str, tag: str) -> str:
        """Extracts content from within the last <tag>...</tag> block."""
        try:
            start_tag = f"<{tag}>"
            end_tag = f"</{tag}>"
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag) : end_pos].strip()
        except ValueError:
            logger.warning(
                f"Could not extract content for tag '{tag}' from text: {text}"
            )
            return ""

    def _execute_tool_with_retry(self, tool, content):
        retry_count = 0
        start_time = time.time()

        while retry_count < self.tool_retry_count:
            try:
                result_text = tool.execute(content)
                if result_text:
                    execution_time = time.time() - start_time
                    return {
                        "success": True,
                        "retry_count": retry_count,
                        "execution_time": execution_time,
                        "result": result_text,
                    }
                else:
                    logger.warning(
                        f"Tool({tool.trigger_tag}) returned empty output. Retrying {retry_count + 1}/{self.tool_retry_count}"
                    )
                    retry_count += 1
            except Exception as e:
                logger.error(
                    f"Tool({tool.trigger_tag}) execution failed. Retrying {retry_count + 1}/{self.tool_retry_count}: {e}"
                )
                retry_count += 1

        execution_time = time.time() - start_time
        logger.warning(
            f"Tool({tool.trigger_tag}) execution failed after {self.tool_retry_count} retries. Appending EOS."
        )
        return {
            "success": False,
            "retry_count": retry_count,
            "execution_time": execution_time,
            "result": "",
        }

    def _calc_entropy(self, logprobs):
        if not logprobs:
            return 0.0
        p_list = [math.exp(l) for l in logprobs]
        entropy = -sum(p * l for p, l in zip(p_list, logprobs))
        return entropy



    def _generate_for_nodes(self, nodes: List[ToolTreeNode], max_len: int, 
                           eos_token_id: int) -> List:
        """Generate tokens for a list of nodes.
        
        Args:
            nodes: List of nodes to generate for
            max_len: Maximum response length
            eos_token_id: EOS token ID
            
        Returns:
            List of generation outputs
        """
        active_prompts = [node.curr_token_ids for node in nodes]
        max_tokens_list = [
            max(1, max_len - (len(node.curr_token_ids) - len(node.prompt_token_ids)))
            for node in nodes
        ]
        max_tokens = max(max_tokens_list)
        
        with self.update_sampling_params(
            n=1,
            stop=self.stop_sequences,
            max_tokens=max_tokens,
            detokenize=True,
            logprobs=self.logprobs,
        ):
            outputs = self.inference_engine.generate(
                prompt_token_ids=active_prompts,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
        
        return outputs

    def _process_node_outputs(self, nodes: List[ToolTreeNode], outputs: List,
                             eos_token_id: int, max_len: int, 
                             tool_metrics: Dict, calls_per_tool: Counter,
                             success_per_tool: Counter) -> tuple:
        """Process generation outputs and update node states.
        
        Args:
            nodes: List of nodes that were generated for
            outputs: Generation outputs from vLLM
            eos_token_id: EOS token ID
            max_len: Maximum response length
            tool_metrics: Metrics dict to update
            calls_per_tool: Counter for calls per tool
            success_per_tool: Counter for successful calls (not used here but kept for consistency)
            
        Returns:
            Tuple of (tool_requests dict, next_nodes list)
        """
        tool_requests = {tag: [] for tag in self.tools}
        next_nodes = []
        
        for i, node in enumerate(nodes):
            output = outputs[i]
            generated_tokens = output.outputs[0].token_ids
            
            # Update current node with generated tokens
            node.curr_token_ids.extend(generated_tokens)
            node.result_mask.extend([1] * len(generated_tokens))
            
            # Record the length of this rollout (last modification to curr_token_ids)
            node.last_rollout_length = len(generated_tokens)
            
            # Calculate average log probability of generated tokens
            # This reflects the model's confidence in the generated sequence
            logprobs = []
            plugin_entropies = []
            tokens = output.outputs[0].token_ids
            for j in range(len(tokens)):
                try:
                    logprob_info = output.outputs[0].logprobs[j]
                    # Get the logprob of the actual generated token
                    token_id = tokens[j]
                    if token_id in logprob_info:
                        logprobs.append(logprob_info[token_id].logprob)
                    else:
                        # If token not in logprob_info, use the first available logprob
                        logprobs.append(list(logprob_info.values())[0].logprob)
                    if self.entropy_estimator == "plugin":
                        plugin_entropies.append(
                            self._calc_entropy([lp.logprob for lp in logprob_info.values()]))
                except Exception as e:
                    logger.warning(f"Failed to get logprob for token {j}: {e}")
                    continue

            # Calculate average log probability (higher is better, closer to 0)
            if self.entropy_estimator == "plugin" and plugin_entropies:
                # Distributional entropy averaged over the segment, computed with the
                # existing _calc_entropy from the top-k logprobs vLLM already returns
                # (self.logprobs = 10). This is the quantity the paper's first named
                # contribution -- "Entropy-Guided Tree Expansion" -- refers to. The
                # default branch below instead averages the surprisal of the SAMPLED
                # tokens, which is an unbiased but high-variance one-sample estimate of
                # a different thing (path-average entropy of the realised continuation).
                # Truncated at top-k and not renormalised, matching TreeAdv's stated
                # approximation.
                node.entropy = sum(plugin_entropies) / len(plugin_entropies)
            elif logprobs:
                node.entropy = -sum(logprobs) / len(logprobs)
            
            # Store initial entropy
            if node.initial_entropy == 0.0 and node.entropy > 0.0:
                node.initial_entropy = node.entropy
            
            # Check finish reason
            finish_reason = output.outputs[0].finish_reason
            stop_reason = output.outputs[0].stop_reason
            is_tool_call = finish_reason == "stop" and stop_reason in self.stop_sequences
            
            if is_tool_call:
                tag = stop_reason.strip("</>")
                if node.call_counter < self.tool_call_limit:
                    node.call_counter += 1
                    full_text = self.tokenizer.decode(node.curr_token_ids)
                    content = self._extract_content(full_text, tag)
                    if content:
                        tool_requests[tag].append({"node": node, "content": content})
                        # Update tool call statistics here
                        tool_metrics["tools/total_calls"] += 1
                        calls_per_tool[tag] += 1
                    else:
                        # No valid content, mark as leaf
                        node.curr_token_ids.append(eos_token_id)
                        node.result_mask.append(1)
                        node.is_active = False
                        node.is_leaf = True
                else:
                    # Tool call limit reached - add warning message and continue generation
                    warning_text = " I have reached the tool call limit and cannot make further tool calls. Let me provide the final answer based on the information I have gathered so far. \n\n"
                    warning_tokens = self.tokenizer.encode(warning_text, add_special_tokens=False)
                    node.curr_token_ids.extend(warning_tokens)
                    node.result_mask.extend([0] * len(warning_tokens))  # 0 because this is system-inserted, not model-generated
                    tool_metrics["tools/call_limit_reached_count"] += 1
                    # Add to next_nodes to continue generation instead of marking as leaf
                    if len(node.curr_token_ids) - len(node.prompt_token_ids) < max_len:
                        next_nodes.append(node)
                    else:
                        node.is_active = False
                        node.is_leaf = True
            
            elif finish_reason == "length":
                response_len = len(node.curr_token_ids) - len(node.prompt_token_ids)
                if response_len >= max_len:
                    node.is_active = False
                    node.is_leaf = True
                else:
                    # Continue generation
                    next_nodes.append(node)
            
            elif finish_reason == "stop":
                # EOS reached, mark as leaf
                node.is_active = False
                node.is_leaf = True
        
        return tool_requests, next_nodes

    def _execute_tools(self, tool_requests: Dict, 
                      tool_metrics: Dict,
                      calls_per_tool: Counter,
                      success_per_tool: Counter,
                      total_time_per_tool: Counter) -> Dict:
        """Execute tool requests in parallel.
        
        Args:
            tool_requests: Dict mapping tool tags to request lists
            tool_metrics: Metrics dict to update
            calls_per_tool: Counter for calls per tool
            success_per_tool: Counter for successful calls
            total_time_per_tool: Counter for execution time per tool
            
        Returns:
            Dict mapping nodes to their tool execution results
        """
        tool_results = {}
        
        if not any(tool_requests.values()):
            return tool_results
        
        total_requests = sum(len(reqs) for reqs in tool_requests.values())
        logger.info(f"Processing {total_requests} tool requests...")
        
        futures = {}
        for tag, requests in tool_requests.items():
            if not requests:
                continue
            tool = self.tools[tag]
            for req in requests:
                future = self.executor.submit(
                    self._execute_tool_with_retry, tool, req["content"]
                )
                futures[future] = {"node": req["node"], "tag": tag}
        
        # Wait for completion and collect results
        for future in concurrent.futures.as_completed(futures):
            fut_info = futures[future]
            parent_node = fut_info["node"]
            tag = fut_info["tag"]
            
            try:
                result = future.result(timeout=self.tool_timeout)
                success = result["success"]
                retry_count = result["retry_count"]
                execution_time = result["execution_time"]
                result_text = result["result"]
                
                # Update metrics (note: total_calls and calls_per_tool are already updated in _process_node_outputs)
                if success:
                    tool_metrics["tools/successful_calls"] += 1
                    success_per_tool[tag] += 1
                else:
                    tool_metrics["tools/failed_calls"] += 1
                    result_text = f"Tool({tag}) returned empty output."
                
                tool_metrics["tools/total_execution_time"] += execution_time
                tool_metrics["tools/max_execution_time"] = max(
                    tool_metrics["tools/max_execution_time"], execution_time
                )
                tool_metrics["tools/total_retries"] += retry_count
                tool_metrics["tools/max_retries"] = max(
                    tool_metrics["tools/max_retries"], retry_count
                )
                total_time_per_tool[tag] += execution_time
                
                # Store result for this node
                tool_results[parent_node] = {
                    "success": success,
                    "result_text": result_text,
                    "tag": tag
                }
                
            except concurrent.futures.TimeoutError:
                logger.error(f"Tool execution timeout for tag {tag}")
                tool_metrics["tools/failed_calls"] += 1
                tool_results[parent_node] = {
                    "success": False,
                    "error": "timeout",
                    "tag": tag
                }
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                tool_metrics["tools/failed_calls"] += 1
                tool_results[parent_node] = {
                    "success": False,
                    "error": str(e),
                    "tag": tag
                }
        
        return tool_results

    def _create_child_nodes_from_tool_results(self, tool_results: Dict) -> List[ToolTreeNode]:
        """Create child nodes from tool execution results.
        
        Args:
            tool_results: Dict mapping parent nodes to their tool results
            
        Returns:
            List of newly created child nodes
        """
        child_nodes = []
        
        for parent_node, result_info in tool_results.items():
            if result_info["success"]:
                # Create child node with tool result
                result_text = result_info["result_text"]
                formatted_result = f" <result>\n{result_text}\n</result>"
                result_tokens = self.tokenizer.encode(formatted_result)
                
                # Check if adding tool result would exceed max_model_len
                # If so, truncate result_tokens to leave room for 100 token margin
                max_model_len = getattr(
                    self.inference_engine.llm_engine.model_config,
                    "max_model_len",
                    8192,
                )
                
                parent_curr_tokens = parent_node.curr_token_ids
                
                if len(parent_curr_tokens) + len(result_tokens) > max_model_len:
                    logger.warning(
                        f"Node {parent_node.node_uid} would exceed max_model_len after tool result, truncating result_tokens"
                    )
                    # Truncate result_tokens to fit, leaving 100 token margin for next generation
                    max_result_length = max_model_len - len(parent_curr_tokens) - 100
                    if max_result_length > 0:
                        result_tokens = result_tokens[:max_result_length]
                    else:
                        # If no room even for truncated result, mark parent as leaf and skip
                        logger.warning(
                            f"Node {parent_node.node_uid} has no room for tool result (max_result_length={max_result_length}), marking as leaf"
                        )
                        parent_node.is_active = False
                        parent_node.is_leaf = True
                        continue
                
                # Create new child node with parent tokens + (possibly truncated) tool result
                child_uid = f"{parent_node.node_uid}_tool{parent_node.call_counter}"
                child_node = ToolTreeNode(
                    tree_uid=parent_node.tree_uid,
                    node_uid=child_uid,
                    prompt_token_ids=parent_node.prompt_token_ids,
                    curr_token_ids=parent_curr_tokens + result_tokens,
                    result_mask=parent_node.result_mask + [0] * len(result_tokens),
                    call_counter=parent_node.call_counter,
                    parent_node=parent_node,
                    is_root=False,
                    is_active=True,
                    is_leaf=False,
                    depth=parent_node.depth + 1,
                    entropy=parent_node.entropy,
                )
                child_node.initial_entropy = parent_node.initial_entropy
                
                parent_node.add_child(child_node)
                parent_node.is_active = False  # Parent is no longer active
                child_nodes.append(child_node)
            else:
                # Tool execution failed, mark parent as leaf
                parent_node.is_active = False
                parent_node.is_leaf = True
        
        return child_nodes

    def _generate_chain_for_nodes(self, initial_nodes: List[ToolTreeNode],
                                 max_len: int, eos_token_id: int,
                                 tool_metrics: Dict, calls_per_tool: Counter,
                                 success_per_tool: Counter, 
                                 total_time_per_tool: Counter,
                                 phase_name: str = ""):
        """Generate complete chains for a list of nodes.
        
        This method handles the iterative generation process: generate tokens,
        process outputs, execute tools, create child nodes, and repeat until
        all nodes are complete.
        
        Args:
            initial_nodes: List of nodes to start generation from
            max_len: Maximum response length
            eos_token_id: EOS token ID
            tool_metrics: Metrics dict to update
            calls_per_tool: Counter for calls per tool
            success_per_tool: Counter for successful calls
            total_time_per_tool: Counter for execution time per tool
            phase_name: Name of the phase for logging
        """
        iteration = 0
        current_nodes = initial_nodes
        
        while current_nodes:
            iteration += 1
            logger.info(f"  {phase_name} Iteration {iteration}: {len(current_nodes)} nodes")
            
            # Generate tokens
            outputs = self._generate_for_nodes(current_nodes, max_len, eos_token_id)
            
            # Process outputs
            tool_requests, next_nodes = self._process_node_outputs(
                current_nodes, outputs, eos_token_id, max_len, tool_metrics,
                calls_per_tool, success_per_tool
            ) #这里的next_nodes包含finish_reason == "length"，未识别出工具调用，但是依然可以往后生成的node
            
            # Execute tools
            tool_results = self._execute_tools(
                tool_requests, tool_metrics, calls_per_tool,
                success_per_tool, total_time_per_tool
            )
            
            # Create child nodes from tool results
            child_nodes = self._create_child_nodes_from_tool_results(tool_results)
            #这里指工具调用之后新增的node
            
            # Filter child nodes by response length (similar to vllm_rollout_with_tools_tree.py)
            # Only keep nodes that haven't exceeded max_len after adding tool results
            valid_child_nodes = []
            for child_node in child_nodes:
                response_len = len(child_node.curr_token_ids) - len(child_node.prompt_token_ids)
                if response_len < max_len:
                    valid_child_nodes.append(child_node)
                else:
                    # Mark as leaf if response length exceeded
                    logger.warning(
                        f"Node {child_node.node_uid} exceeded max_len ({response_len} >= {max_len}) after tool result, marking as leaf"
                    )
                    child_node.is_active = False
                    child_node.is_leaf = True
            
            # Add valid child nodes to next iteration
            next_nodes.extend(valid_child_nodes)
            
            # Update current nodes for next iteration
            current_nodes = next_nodes

    def _calculate_tool_metrics(self, tool_metrics, calls_per_tool, success_per_tool, 
                               total_time_per_tool):
        """Calculate final tool metrics including per-tool statistics.
        
        Args:
            tool_metrics: Base metrics dict
            calls_per_tool: Counter for calls per tool
            success_per_tool: Counter for successful calls per tool
            total_time_per_tool: Counter for total execution time per tool
            
        Returns:
            Dict with all metrics including tool-specific ones
        """
        # Calculate average execution time
        if tool_metrics["tools/total_calls"] > 0:
            tool_metrics["tools/avg_execution_time"] = (
                tool_metrics["tools/total_execution_time"]
                / tool_metrics["tools/total_calls"]
            )

        # Calculate per-tool metrics
        tool_specific_metrics = {}
        for tag in self.tools.keys():
            calls = calls_per_tool[tag]
            if calls > 0:
                tool_specific_metrics[f"tools/{tag}/calls"] = calls
                tool_specific_metrics[f"tools/{tag}/avg_time"] = (
                    total_time_per_tool[tag] / calls
                )
                tool_specific_metrics[f"tools/{tag}/success_rate"] = (
                    success_per_tool[tag] / calls
                )
            else:
                tool_specific_metrics[f"tools/{tag}/calls"] = 0
                tool_specific_metrics[f"tools/{tag}/avg_time"] = 0
                tool_specific_metrics[f"tools/{tag}/success_rate"] = 0

        return {**tool_metrics, **tool_specific_metrics}


    @GPUMemoryLogger(role="vllm rollout spmd with tools", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        print('generating via vllm_rollout_with_tools_tree_offline')
        if vllm_version in ("0.5.4", "0.6.3") and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = self.tokenizer.eos_token_id
        batch_size = input_ids.size(0)

        # Initialize tool call statistics
        tool_metrics = {
            "tools/total_calls": 0,
            "tools/successful_calls": 0,
            "tools/failed_calls": 0,
            "tools/total_execution_time": 0.0,
            "tools/avg_execution_time": 0.0,
            "tools/max_execution_time": 0.0,
            "tools/max_retries": 0,
            "tools/total_retries": 0,
            "tools/call_limit_reached_count": 0,
        }

        calls_per_tool = Counter()
        success_per_tool = Counter()
        total_time_per_tool = Counter()

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)

        # Determine beam_size based on mode
        beam_size = self.beam_size
        initial_rollouts = self.initial_rollouts
        expansion_iterations = self.expansion_iterations
        samples_per_tree = self.samples_per_tree

        if not do_sample:
            kwargs.update({
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,
            })
            beam_size = 1
            initial_rollouts = 1  # No branching in greedy mode
            expansion_iterations = 0
            samples_per_tree = 1
        elif is_validate:
            kwargs.update({
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,
            })
            beam_size = 1
            initial_rollouts = 1  # No branching in validation mode
            expansion_iterations = 0
            samples_per_tree = 1

        # Eval-time decoding knobs for the decoding-control experiment. The late-training
        # collapse is entirely non-emission, and the unanswered rows compress to 1.3% of
        # their size (zlib) with 93% of their 80-char windows duplicated -- i.e. they are
        # degenerate repetition, which is exactly what a repetition penalty targets. The
        # question this answers is whether the failure can be masked at decode time.
        #
        # Note it does NOT answer whether training caused it: the decoder is held constant
        # across checkpoints, and the degeneration rate still quadruples between step 180
        # and step 200, so the change is attributable to the policy either way.
        #
        # The branches above hardcode the greedy protocol (and min_p=0.0), so these are
        # applied afterwards to take precedence. Defaults equal vLLM's own, so an unset
        # config reproduces the baseline protocol byte for byte.
        if is_validate or not do_sample:
            for _cfg_key, _sp_attr, _default in (
                ("val_repetition_penalty", "repetition_penalty", 1.0),
                ("val_frequency_penalty", "frequency_penalty", 0.0),
                ("val_min_p", "min_p", 0.0),
            ):
                try:
                    _v = float(getattr(self.config, _cfg_key))
                except Exception:
                    _v = _default
                if _v != _default:
                    kwargs[_sp_attr] = _v
                    logger.info(f"[decoding-control] {_sp_attr}={_v}")

        # fix oov error
        kwargs["allowed_token_ids"] = list(self.tokenizer.get_vocab().values())
        
        with self.update_sampling_params(**kwargs):
            prompt_token_ids_list = [
                _pre_process_inputs(self.pad_token_id, prompt) for prompt in input_ids
            ]
            
            # Calculate initial rollouts
            if self.enable_dynamic_rollouts:
                raise ValueError(
                    "Dynamic rollouts initialization not supported in offline mode. "
                    "Please set enable_dynamic_rollouts=False in config."
                )
            else:
                # Use initial_rollouts directly for offline mode
                # In offline mode, initial_rollouts determines the number of initial branches from each root
                initial_rollouts_list = [initial_rollouts] * len(prompt_token_ids_list)
                logger.info(
                    f"Using static initial_rollouts={initial_rollouts} initial branches per root for all {len(prompt_token_ids_list)} samples "
                    f"(mode: do_sample={do_sample}, is_validate={is_validate})"
                )
            
            logger.info(f"Final calculated initial_rollouts: {initial_rollouts_list}")
            max_len = self.config.response_length
            
            # ===== PHASE 1: BUILD INITIAL TREES =====
            logger.info("=" * 60)
            logger.info("PHASE 1: BUILD INITIAL TREE STRUCTURES")
            logger.info("=" * 60)
            
            root_nodes = self._build_initial_trees(prompt_token_ids_list, initial_rollouts_list)
            total_roots = len(root_nodes)
            total_initial_branches = sum(len(root.child_nodes) for root in root_nodes)
            avg_branches_per_root = total_initial_branches / total_roots if total_roots > 0 else 0
            logger.info(f"Created {total_roots} root nodes with {total_initial_branches} initial branches total "
                       f"({avg_branches_per_root:.1f} branches/root avg)")
            
            # ===== PHASE 2: GENERATE INITIAL ACTION CHAINS =====
            logger.info("=" * 60)
            logger.info("PHASE 2: GENERATE INITIAL ACTION CHAINS")
            logger.info("=" * 60)
            
            # Collect all initial branches from roots
            initial_branches = []
            for root in root_nodes:
                initial_branches.extend(root.child_nodes)
            
            # Generate complete chains for all initial branches
            self._generate_chain_for_nodes(
                initial_nodes=initial_branches,
                max_len=max_len,
                eos_token_id=eos_token_id,
                tool_metrics=tool_metrics,
                calls_per_tool=calls_per_tool,
                success_per_tool=success_per_tool,
                total_time_per_tool=total_time_per_tool,
                phase_name="Phase 2"
            )
            
            logger.info("PHASE 2 COMPLETED: Initial action chains generated")
            
            # ===== PHASE 3: TREE EXPANSION =====
            logger.info("=" * 60)
            logger.info("PHASE 3: TREE EXPANSION")
            logger.info("=" * 60)
            
            for exp_iter in range(expansion_iterations):
                logger.info(f"=== Expansion Iteration {exp_iter + 1}/{expansion_iterations} ===")
                logger.info(f"Using expansion mode: {self.expansion_mode}")
                
                # Select nodes for expansion from each tree
                expansion_nodes = []
                
                for tree_idx, root in enumerate(root_nodes):
                    # Sample expansion nodes with entropy-based probability
                    # Penalty is now calculated per-node based on existing children
                    nodes_to_expand = root.sample_expansion_nodes(
                        beam_size, 
                        mode=self.expansion_mode,
                        entropy_weight=self.entropy_weight,
                        branch_probability=self.branch_probability,
                        score_norm=self.expansion_score_norm,
                        penalty_lambda=(self.expansion_penalty_lambda / max(1, self.num_branches)
                                        if self.expansion_penalty_per_branch
                                        else self.expansion_penalty_lambda),
                    )
                    record_selection(root, exp_iter, nodes_to_expand)
                    
                    if nodes_to_expand:
                        expansion_nodes.extend(nodes_to_expand)
                
                # Log entropy information if using entropy mode
                if self.expansion_mode == 'entropy' and expansion_nodes:
                    logger.info(f"Selected {len(expansion_nodes)} nodes with entropy values:")
                    for node in expansion_nodes[:5]:  # Log first 5 nodes
                        num_children = len(node.child_nodes)
                        logger.info(f"  Node {node.node_uid}: entropy={node.entropy:.4f}, initial_entropy={node.initial_entropy:.4f}, existing_children={num_children}")
                
                if not expansion_nodes:
                    logger.info("No nodes available for expansion")
                    break
                
                # Create new branches for each expansion node
                new_branches = []
                branches_per_node_list = []
                for node in expansion_nodes:
                    # Create num_branches branches for each selected node
                    # Currently set to 1, but can be configured via parameter in the future
                    num_branches = self.num_branches
                    branches_per_node_list.append(num_branches)
                    
                    # New branches should be created from the parent node, not the sampled node itself
                    # This makes the new branch a sibling of the sampled node
                    parent_node = node.parent_node
                    
                    if parent_node is None:
                        # If sampled node is root, create branch from root
                        logger.warning(f"Sampled node {node.node_uid} is root, creating branch from root")
                        parent_node = node
                    
                    for branch_idx in range(num_branches):
                        # Use parent's existing child count to generate unique branch_id
                        branch_id = len(parent_node.child_nodes) + branch_idx
                        new_branch = parent_node.create_branch(branch_id)
                        
                        # Copy the sampled node's state, but remove the last rollout content
                        # This preserves all tool call results while allowing re-generation of the last rollout
                        if node.last_rollout_length > 0:
                            # Remove last_rollout_length tokens from the sampled node's state
                            rollback_length = node.last_rollout_length
                            new_branch.curr_token_ids = node.curr_token_ids[:-rollback_length]
                            new_branch.result_mask = node.result_mask[:-rollback_length]
                        else:
                            # If no rollout to remove, just copy the sampled node's state
                            new_branch.curr_token_ids = node.curr_token_ids.copy()
                            new_branch.result_mask = node.result_mask.copy()
                        
                        # Inherit other properties from sampled node to maintain consistency
                        new_branch.call_counter = max(0, node.call_counter - 1)
                        new_branch.depth = node.depth
                        new_branch.entropy = node.entropy
                        new_branch.initial_entropy = node.initial_entropy
                        
                        # The post-selection correction walks up from each leaf looking for the
                        # ancestor that was SELECTED, and must stop at a fresh draw. Set the
                        # marker here rather than relying on mark_new_branch(), which returns
                        # early when TREE_STATS_PATH is unset -- the correction's semantics must
                        # not depend on whether instrumentation happens to be enabled.
                        new_branch._created_iter = exp_iter
                        new_branches.append(new_branch)
                        mark_new_branch(new_branch, node, exp_iter, branch_idx)
                
                # Log with average branches per node
                avg_branches = sum(branches_per_node_list) / len(branches_per_node_list) if branches_per_node_list else 0
                logger.info(f"Created {len(new_branches)} new branches from {len(expansion_nodes)} nodes (num_branches={num_branches}, avg_branches_per_node={avg_branches:.1f})")
                
                # Generate complete chains for new branches
                self._generate_chain_for_nodes(
                    initial_nodes=new_branches,
                    max_len=max_len,
                    eos_token_id=eos_token_id,
                    tool_metrics=tool_metrics,
                    calls_per_tool=calls_per_tool,
                    success_per_tool=success_per_tool,
                    total_time_per_tool=total_time_per_tool,
                    phase_name=f"Phase 3 (Expansion {exp_iter+1})"
                )
            
            logger.info("PHASE 3 COMPLETED: Tree expansion finished")
            
            # ===== PHASE 4: COLLECT FINAL OUTPUTS =====
            logger.info("=" * 60)
            logger.info("PHASE 4: COLLECT FINAL OUTPUTS")
            logger.info("=" * 60)
            
            output_sequences = []
            output_result_masks = []
            collected_leaf_depths = []
            collected_leaf_nodes = []  # Store leaf node references for reward assignment
            
            # Now each root corresponds to one input sample (tree_idx == sample_id)
            for sample_id, root in enumerate(root_nodes):
                # Sample samples_per_tree leaves from this tree
                sampled_leaves = root.sample_leaves(samples_per_tree)
                
                # Log tree statistics
                all_leaves = root.get_all_leaves()
                logger.info(f"Sample {sample_id}: {len(all_leaves)} total leaves, sampling {len(sampled_leaves)}")

                for leaf in sampled_leaves:
                    response_tokens = leaf.curr_token_ids[len(leaf.prompt_token_ids):]
                    # Extract corresponding result_mask for response part
                    response_mask = leaf.result_mask.copy()
                    
                    # Ensure response doesn't exceed max_len
                    if len(response_tokens) > max_len:
                        response_tokens = response_tokens[:max_len]
                        response_mask = response_mask[:max_len]
                    
                    output_sequences.append(response_tokens)
                    output_result_masks.append(response_mask)
                    collected_leaf_depths.append(leaf.depth)
                    collected_leaf_nodes.append(leaf)  # Save leaf reference
            
            logger.info(f"Collected {len(output_sequences)} output sequences (samples_per_tree={samples_per_tree})")
            
            # Log depth statistics for collected leaf nodes
            if collected_leaf_depths:
                logger.info(f"Collected leaf node depths: {collected_leaf_depths}")
                logger.info(f"Depth statistics - min: {min(collected_leaf_depths)}, max: {max(collected_leaf_depths)}, avg: {sum(collected_leaf_depths)/len(collected_leaf_depths):.2f}")
            
            # Validate output count
            # Each input sample has 1 tree, each tree samples samples_per_tree outputs
            expected_output_count = batch_size * samples_per_tree
            actual_output_count = len(output_sequences)
            if actual_output_count != expected_output_count:
                logger.warning(
                    f"Output count mismatch: expected {expected_output_count} "
                    f"(batch_size={batch_size} * samples_per_tree={samples_per_tree}), "
                    f"got {actual_output_count}"
                )
            else:
                logger.info(f"Output count validated: {actual_output_count} sequences collected")
            
            # ===== PHASE 5: PAD AND STACK OUTPUTS =====
            logger.info("=" * 60)
            logger.info("PHASE 5: PAD AND STACK OUTPUTS")
            logger.info("=" * 60)
            
            padded_response_list = []
            padded_result_mask_list = []
            
            for output_ids, result_mask in zip(output_sequences, output_result_masks):
                # Ensure lengths match
                if len(output_ids) != len(result_mask):
                    min_len = min(len(output_ids), len(result_mask))
                    output_ids = output_ids[:min_len]
                    result_mask = result_mask[:min_len]
                
                response = torch.tensor(output_ids)
                response = pad_sequence_to_length(
                    response, self.config.response_length, self.pad_token_id
                )

                result_mask_tensor = torch.tensor(result_mask)
                result_mask_tensor = pad_sequence_to_length(
                    result_mask_tensor, self.config.response_length, 0
                )

                padded_response_list.append(response)
                padded_result_mask_list.append(result_mask_tensor)

            response = torch.stack(padded_response_list, dim=0).to(input_ids.device)
            loss_mask = torch.stack(padded_result_mask_list, dim=0).to(input_ids.device)
            
            # Repeat input_ids if needed
            non_tensor_batch = deepcopy(prompts.non_tensor_batch)
            
            # Only repeat if we have the expected number of outputs
            # Each input sample produces samples_per_tree outputs (from 1 tree)
            if actual_output_count == expected_output_count and samples_per_tree > 1 and do_sample:
                input_ids = _repeat_interleave(input_ids, samples_per_tree)
                attention_mask = _repeat_interleave(attention_mask, samples_per_tree)
                position_ids = _repeat_interleave(position_ids, samples_per_tree)
                if non_tensor_batch:
                    for key, value in non_tensor_batch.items():
                        if isinstance(value, np.ndarray):
                            non_tensor_batch[key] = np.repeat(
                                value, samples_per_tree, axis=0
                            )
                        elif isinstance(value, list):
                            non_tensor_batch[key] = [
                                item for item in value for _ in range(samples_per_tree)
                            ]
                logger.info(f"Repeated input tensors by samples_per_tree={samples_per_tree}")
            elif actual_output_count != expected_output_count:
                logger.error(
                    f"Cannot proceed with mismatched output count. "
                    f"Expected {expected_output_count}, got {actual_output_count}. "
                    f"This may cause downstream errors."
                )

            final_batch_size = input_ids.size(0)
            seq = torch.cat([input_ids, response], dim=-1)

            response_length = response.size(1)
            delta_position_id = (
                torch.arange(1, response_length + 1, device=position_ids.device)
                .unsqueeze(0)
                .expand(final_batch_size, -1)
            )

            if position_ids.dim() == 3:  # for RoPE scaling like qwen2vl mrope
                delta_position_id = delta_position_id.view(
                    final_batch_size, 1, -1
                ).expand(final_batch_size, position_ids.size(1), -1)
                response_position_ids = (
                    position_ids[..., -1:].expand(-1, position_ids.size(1), -1)
                    + delta_position_id
                )
            else:
                response_position_ids = position_ids[..., -1:] + delta_position_id

            final_position_ids = torch.cat(
                [position_ids, response_position_ids], dim=-1
            )

            response_attention_mask = get_response_mask(
                response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
            )
            final_attention_mask = torch.cat(
                (attention_mask, response_attention_mask), dim=-1
            )

            loss_mask = loss_mask * response_attention_mask

            # Calculate all metrics
            all_metrics = self._calculate_tool_metrics(
                tool_metrics, calls_per_tool, success_per_tool, total_time_per_tool
            )

            batch = TensorDict(
                {
                    "prompts": input_ids,
                    "responses": response,
                    "input_ids": seq,
                    "attention_mask": final_attention_mask,
                    "loss_mask": loss_mask,
                    "position_ids": final_position_ids,
                },
                batch_size=final_batch_size,
            )

        if vllm_version in ("0.5.4", "0.6.3") and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        # Add metrics to meta_info
        meta_info = deepcopy(prompts.meta_info) if prompts.meta_info else {}
        meta_info["metrics"] = all_metrics

        data_proto = DataProto(
            batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info
        )
        
        # ===== PHASE 6: COMPUTE REWARDS AND ADVANTAGES =====
        # Only compute rewards and advantages when not in validation mode
        if not is_validate:
            logger.info("=" * 60)
            logger.info("PHASE 6: COMPUTE REWARDS AND ADVANTAGES")
            logger.info("=" * 60)
            
            # Compute reward model score
            logger.info("Computing reward scores...")
            reward_tensor, reward_extra_infos_dict = compute_reward(data_proto, self.reward_fn)
            
            # Add reward extra info to non_tensor_batch
            if reward_extra_infos_dict:
                data_proto.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
            
            logger.info("Assigning reward values to leaf nodes...")
            for i, leaf_node in enumerate(collected_leaf_nodes):
                # Get valid response length from attention mask
                prompt_length = data_proto.batch["prompts"][i].shape[-1]
                valid_response_length = data_proto.batch["attention_mask"][i, prompt_length:].sum().item()
                
                if valid_response_length > 0:
                    # Get reward value at the last valid position (similar to naive.py)
                    leaf_value = reward_tensor[i, valid_response_length - 1].item()
                    leaf_node.value = leaf_value
                    logger.debug(f"Leaf {leaf_node.node_uid}: assigned value={leaf_value:.4f} from position {valid_response_length - 1}")
                else:
                    # No valid tokens, set value to 0
                    leaf_node.value = 0.0
                    logger.warning(f"Leaf {leaf_node.node_uid}: no valid tokens, set value=0.0")
            
            logger.info(f"Assigned reward values to {len(collected_leaf_nodes)} leaf nodes")

            uid = np.array([str(uuid.uuid4()) for _ in range(len(root_nodes))], dtype=object)
            data_proto.non_tensor_batch["uid"] = np.repeat(uid, samples_per_tree, axis=0)

            if self.node_adv_mode=='vanilla':
                print("Computing token-level rewards and advantages (node_adv_mode == vanilla)...")
                data_proto.batch["token_level_scores"] = reward_tensor
                # Compute token_level_rewards (apply KL penalty if configured)
                if self.use_kl_in_reward and self.kl_ctrl is not None:
                    logger.info("Applying KL penalty to rewards...")
                    # Note: This requires old_log_probs and ref_log_prob to be computed
                    # These should be computed by the trainer before calling this method
                    # For now, we'll just use token_level_scores as token_level_rewards
                    if "old_log_probs" in data_proto.batch and "ref_log_prob" in data_proto.batch:
                        data_proto, kl_metrics = apply_kl_penalty(
                            data_proto, 
                            kl_ctrl=self.kl_ctrl, 
                            kl_penalty=self.kl_penalty,
                            multi_turn=False
                        )
                        all_metrics.update(kl_metrics)
                    else:
                        logger.warning("old_log_probs or ref_log_prob not found in batch, skipping KL penalty")
                        data_proto.batch["token_level_rewards"] = data_proto.batch["token_level_scores"]
                else:
                    data_proto.batch["token_level_rewards"] = data_proto.batch["token_level_scores"]
                    
                # Compute advantages and returns
                logger.info(f"Computing advantages using {self.adv_estimator}...")
                data_proto = compute_advantage(
                    data_proto,
                    adv_estimator=self.adv_estimator,
                    gamma=self.gamma,
                    lam=self.lam,
                    num_repeat=self.samples_per_tree,
                    multi_turn=False,
                    norm_adv_by_std_in_grpo=self.norm_adv_by_std_in_grpo,
                )
            else:
                # Process reward mode: compute node-level rewards
                logger.info("Computing process rewards (node_adv_mode != vanilla)...")
                stash_raw_leaf_scores(root_nodes)
                
                # Step 1: Normalize leaf node scores if configured

                if self.leaf_value_norm:
                    print("Normalizing leaf node scores within each tree...")
                    epsilon = 1e-6
                    # Directly iterate over root_nodes, each root corresponds to one tree
                    for root in root_nodes:
                        tree_uid = root.tree_uid
                        tree_leaves = root.get_all_leaves()
                        
                        # After sample_leaves, each tree should have exactly samples_per_tree leaves
                        assert len(tree_leaves) == samples_per_tree, \
                            f"Tree {tree_uid}: expected {samples_per_tree} leaves after sampling, got {len(tree_leaves)}"
                        
                        # Collect leaf values
                        leaf_values = [leaf.value for leaf in tree_leaves if leaf.value is not None]
                        
                        # Calculate mean and std
                        leaf_values_array = np.array(leaf_values)
                        mean_value = np.mean(leaf_values_array)
                        std_value = np.std(leaf_values_array)
                        
                        # Normalize each leaf value: (value - mean) / (std + epsilon)
                        for leaf in tree_leaves:
                            if leaf.value is not None:
                                leaf.value = (leaf.value - mean_value) / (std_value + epsilon)
                        
                        logger.info(f"Tree {tree_uid}: normalized {len(tree_leaves)} leaf nodes (mean={mean_value:.4f}, std={std_value:.4f})")
                
                # Step 1b: within-outcome-group cost tie-break.
                #
                # Measured motivation: 86-96% of leaves sit in an outcome group of size >=2
                # (mean group size ~11 of 22), and at a converged policy 52-66% of whole
                # trees have every leaf scored identically. Every leaf in such a group
                # receives exactly the same group-relative advantage, so the tree spends
                # ~11 rollouts to produce one distinguishable unit of credit -- and a fully
                # tied tree contributes no gradient at all.
                #
                # Inside a group the outcome is by definition identical, but the COST is
                # not: 81-92% of groups differ in tool-call count, with a within-group sd
                # of ~1.2 calls. So add a within-group, zero-mean, standardised cost
                # contrast: among equally-scoring trajectories, prefer the cheaper one.
                #
                # Safety is structural, not a heuristic: the term sums to zero inside every
                # group, and the primary advantage is constant inside a group, so no
                # group's mean advantage moves. The between-group signal -- the entire
                # existing objective -- is preserved exactly. Gated to groups whose cost
                # actually varies, and disabled at weight 0, which reproduces the previous
                # behaviour byte for byte.
                if self.cost_tiebreak_weight > 0:
                    for root in root_nodes:
                        groups = {}
                        for lf in root.get_all_leaves():
                            if lf.value is None:
                                continue
                            groups.setdefault(round(float(lf.value), 6), []).append(lf)
                        for _, g in groups.items():
                            if len(g) < 2:
                                continue
                            costs = np.array([float(l.depth) for l in g])
                            sd = costs.std()
                            if sd < 1e-9:
                                continue
                            adj = -self.cost_tiebreak_weight * (costs - costs.mean()) / sd
                            for l, a in zip(g, adj):
                                l.value = float(l.value) + float(a)

                # Step 1c: post-selection (order-statistic) correction.
                #
                # Estimate a2 from THIS batch first and stash it for the NEXT step, before
                # any correction touches the values. Correcting with a coefficient fitted
                # on the values being corrected would be self-referential.
                _a2_next, _a2_n = _candidate_level_a2(root_nodes)

                # A node is expanded because its own realised statistic ranked in the top
                # `beam_size` of its event, so its value is a SELECTED order statistic while
                # a freshly drawn sibling is not. If the selection statistic S and the
                # outcome R are jointly roughly normal with correlation a2, then
                #
                #     E[R | rank r of n] - E[R] = a2 * sd(R) * E[Z_(r:n)]
                #
                # Step 1 has already z-scored leaf values within each tree, so on this scale
                # sd(R) == 1 and the correction is just `a2_hat * E[Z_(r:n)]`. E[Z] uses
                # Blom's approximation, which matches Monte Carlo to <0.008 over the ranks
                # and candidate counts this configuration produces. `a2_hat` comes from the
                # PREVIOUS step only, so the correction is causal and needs no lookahead.
                #
                # Applied once per leaf, using the deepest selected ancestor (or the leaf
                # itself). Freshly created branches were never selected and are left alone,
                # which is the whole point -- they are the unconditioned draws.
                if self.postsel_correction_weight > 0 and _A2_STATE["a2"] is not None:
                    a2 = _A2_STATE["a2"]
                    w = self.postsel_correction_weight
                    n_corr = 0
                    for root in root_nodes:
                        for leaf in root.get_all_leaves():
                            if leaf.value is None:
                                continue
                            cur, sel = leaf, None
                            while cur is not None:
                                if getattr(cur, "_created_iter", None) is not None:
                                    break          # fresh draw: not a selected unit
                                if getattr(cur, "_sel_rank", None) is not None:
                                    sel = cur
                                    break
                                cur = cur.parent_node
                            if sel is None:
                                continue
                            ez = _expected_order_stat(sel._sel_rank, sel._sel_ncand)
                            leaf.value = float(leaf.value) - w * a2 * ez
                            n_corr += 1
                    logger.info(f"post-selection correction: a2={a2:+.4f} weight={w} "
                                f"corrected {n_corr} leaves")

                if _a2_next is not None:
                    _A2_STATE["a2"], _A2_STATE["n"] = _a2_next, _a2_n
                    logger.info(f"post-selection a2 for next step: {_a2_next:+.4f} "
                                f"(over {_a2_n} candidates)")

                # Step 2: Compute node values based on node_value_mode
                if self.node_value_mode in ['child_mean', 'child_softmax']:
                    print(f"Computing node values using {self.node_value_mode} mode...")
                    for root in root_nodes:
                        root.compute_value_from_children(mode=self.node_value_mode)
                        logger.info(f"Tree {root.tree_uid}: computed node values using {self.node_value_mode} (root value={root.value:.4f})")
                elif self.node_value_mode=='leaf_mean':
                    print("Computing node values using leaf_mean mode...")
                    
                    # Compute values for all trees by averaging leaf node values
                    for root in root_nodes:
                        # Get all non-leaf nodes in the tree (including root)
                        all_non_leaf_nodes = root.get_non_leaf_nodes()
                        
                        # Compute value for each non-leaf node based on its descendant leaves
                        for node in all_non_leaf_nodes:
                            # Get all leaf nodes in this node's subtree
                            leaf_nodes = node.get_all_leaves()
                            # Calculate mean of leaf values
                            leaf_values = [leaf.value for leaf in leaf_nodes if leaf.value is not None]
                            node.value = np.mean(leaf_values) if leaf_values else 0.0

                else:
                    raise ValueError(f"Unsupported node_value_mode: {self.node_value_mode}")
                
                # step 3: Compute node advantages based on node_adv_mode
                if self.node_adv_mode == 'node_value':
                    print("Computing node advantages using node_value mode...")
                    # Directly use value as advantage
                    for root in root_nodes:
                        all_nodes = [root] + root.get_subtree_nodes()
                        for node in all_nodes:
                            node.advantage = node.value
                elif self.node_adv_mode == 'diff_parent':
                    print("Computing node advantages using diff_parent mode...")
                    # Use node value minus parent value as advantage
                    for root in root_nodes:
                        # Root node has no parent, set advantage to its own value
                        root.advantage = root.value
                        
                        # For all other nodes, compute advantage as value - parent_value
                        all_descendants = root.get_subtree_nodes()
                        for node in all_descendants:
                            node.advantage = node.value - node.parent_node.value
                elif self.node_adv_mode == 'diff_global':
                    print("Computing node advantages using diff_global mode...")
                    # Use node value minus root value as advantage
                    for root in root_nodes:
                        # Root node has no parent, set advantage to its own value
                        root.advantage = root.value
                        
                        # For all other nodes, compute advantage as value - parent_value
                        all_descendants = root.get_subtree_nodes()
                        for node in all_descendants:
                            node.advantage = node.value - root.value
                elif self.node_adv_mode == 'diff_localglobal':
                    print("Computing node advantages using diff_localglobal mode...")
                    # Use node value minus average of parent and root values as advantage
                    for root in root_nodes:
                        # Root node has no parent, set advantage to its own value
                        root.advantage = root.value
                        
                        # For all other nodes, compute advantage as value - parent_value
                        all_descendants = root.get_subtree_nodes()
                        for node in all_descendants:
                            node.advantage = (node.value - root.value) + (node.value - node.parent_node.value)
                else:
                    raise ValueError(f"Unsupported node_adv_mode: {self.node_adv_mode}")

                dump_tree_stats(
                    root_nodes,
                    os.environ.get("TREE_STATS_PATH", ""),
                    step=int(os.environ.get("TREE_STATS_STEP", "-1")),
                    node_value_mode=self.node_value_mode,
                    node_adv_mode=self.node_adv_mode,
                )
                

                # step 4: Compute token-level scores and advantages from leaf nodes
                print("Computing token-level scores and advantages from leaf nodes...")
                
                # Initialize token-level tensors with zeros
                batch_size = len(collected_leaf_nodes)
                response_length = data_proto.batch["responses"].shape[1]
                token_level_scores = torch.zeros((batch_size, response_length), dtype=torch.float32, device=data_proto.batch["responses"].device)
                token_level_advantages = torch.zeros((batch_size, response_length), dtype=torch.float32, device=data_proto.batch["responses"].device)
                
                # For each leaf node, trace back to root and assign values to token segments
                for i, leaf_node in enumerate(collected_leaf_nodes):
                    # Trace path from leaf to root
                    path_nodes = []
                    current_node = leaf_node
                    while current_node is not None:
                        path_nodes.append(current_node)
                        current_node = current_node.parent_node
                    
                    # Reverse to get root-to-leaf order
                    path_nodes.reverse()
                    
                    # For each node in the path (except root), assign value to its incremental tokens
                    for j in range(1, len(path_nodes)):
                        node = path_nodes[j]
                        parent_node = path_nodes[j - 1]
                        
                        # Calculate token range for this node's contribution
                        parent_len = len(parent_node.curr_token_ids) - len(parent_node.prompt_token_ids)
                        node_len = len(node.curr_token_ids) - len(node.prompt_token_ids)
                        
                        # The incremental tokens are from parent_len to node_len
                        if node_len > parent_len:
                            start_idx = parent_len
                            end_idx = min(node_len, response_length)
                            
                            # Assign node's value and advantage to these tokens
                            token_level_scores[i, start_idx:end_idx] = node.value
                            token_level_advantages[i, start_idx:end_idx] = node.advantage
                            
                            logger.debug(f"Leaf {i}, Node {node.node_uid}: assigned value={node.value:.4f}, adv={node.advantage:.4f} to tokens [{start_idx}:{end_idx}]")

                logger.info(f"Computed token-level scores and advantages for {batch_size} sequences")
                
                # Assign to data_proto
                data_proto.batch["token_level_scores"] = reward_tensor  # use reward_tensor to show in log
                data_proto.batch["token_level_rewards"] = token_level_scores  # Initially same as scores
                data_proto.batch["advantages"] = token_level_advantages
                data_proto.batch["returns"] = token_level_advantages  # Use advantages as returns for now
            
            
            logger.info("PHASE 6 COMPLETED: Rewards and advantages computed")
            logger.info(f"  token_level_scores shape: {data_proto.batch['token_level_scores'].shape}")
            logger.info(f"  token_level_rewards shape: {data_proto.batch['token_level_rewards'].shape}")
            logger.info(f"  advantages shape: {data_proto.batch['advantages'].shape}")
            logger.info(f"  returns shape: {data_proto.batch['returns'].shape}")
        elif is_validate:
            logger.info("=" * 60)
            logger.info("PHASE 6: SKIPPED (validation mode)")
            logger.info("=" * 60)

        logger.info("=" * 60)
        logger.info("GENERATION COMPLETED")
        logger.info("=" * 60)

        return data_proto
