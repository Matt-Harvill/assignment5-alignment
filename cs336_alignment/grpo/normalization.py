"""
Problem (compute_group_normalized_rewards): Group normalization (2 points)
Deliverable: Implement a method compute_group_normalized_rewards that calculates raw
rewards for each rollout response, normalizes them within their groups, and returns both the
normalized and raw rewards along with any metadata you think is useful.
The following interface is recommended:
def compute_group_normalized_rewards(
reward_fn,
rollout_responses,
repeated_ground_truths,
group_size,
advantage_eps,
normalize_by_std,
):
Compute rewards for each group of rollout responses, normalized by the group size.
Args:
reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
the ground truths, producing a dict with keys "reward", "format_reward", and
"answer_reward".
rollout_responses: list[str] Rollouts from the policy. The length of this list is
rollout_batch_size = n_prompts_per_rollout_batch * group_size.
repeated_ground_truths: list[str] The ground truths for the examples. The length of this
list is rollout_batch_size, because the ground truth for each example is repeated
group_size times.
group_size: int Number of responses per question (group).
advantage_eps: float Small constant to avoid division by zero in normalization.
normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise
subtract only the group mean.
Returns:
tuple[torch.Tensor, torch.Tensor, dict[str, float]].
advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout
response.
22
raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout
response.
metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
To test your code, implement [adapters.run_compute_group_normalized_rewards]. Then,
run the test with uv run pytest -k test_compute_group_normalized_rewards and make
sure your implementation passes it.
"""

from collections.abc import Callable
import torch


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],  # length is rollout_batch_size = n_prompts_per_rollout_batch * group_size
    repeated_ground_truths: list[str],  # length is also rollout_batch_size
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[
    torch.Tensor, torch.Tensor, dict[str, float]
]:  # advantages (rollout_batch_size,), raw_rewards (same), metadata
    """
    Compute rewards for each group of rollout responses, normalized by the group size.
    """

    assert (
        len(rollout_responses) == len(repeated_ground_truths)
    ), f"Mismatch: rollout_responses length {len(rollout_responses)} != repeated_ground_truths length {len(repeated_ground_truths)}"

    rollout_batch_size = len(rollout_responses)
    assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"

    n_prompts_pre_rollout_batch = rollout_batch_size // group_size

    advantages: torch.Tensor = torch.empty(size=(rollout_batch_size,))
    raw_rewards: torch.Tensor = torch.empty(size=(rollout_batch_size,))

    # Compute rewards and advantages
    for group_index in range(n_prompts_pre_rollout_batch):
        # Create new tensor for storing group rewards
        group_rewards: torch.Tensor = torch.zeros(
            group_size,
        )

        for sample_index in range(group_size):
            index = (group_index * group_size) + sample_index

            reward = reward_fn(rollout_responses[index], repeated_ground_truths[index])

            group_rewards[sample_index] = reward["answer_reward"]

        # Calculate advantages based on group stats
        group_std, group_mean = torch.std_mean(group_rewards)
        group_advantages = group_rewards - group_mean
        if normalize_by_std:
            group_advantages /= group_std + advantage_eps

        # Update return variables
        raw_rewards[group_index * group_size : (group_index + 1) * group_size] = group_rewards
        advantages[group_index * group_size : (group_index + 1) * group_size] = group_advantages

    # Return metadata too (empty for now)
    metadata: dict[str, float] = {}

    return advantages, raw_rewards, metadata
