import torch
from typing import Literal


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return -policy_log_probs * raw_rewards_or_advantages


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log
        probs from the policy being trained.
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs
        from the old policy.
        cliprange: float Clip parameter ε (e.g. 0.2).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped
        loss.
        metadata dict containing whatever you want to log. We suggest logging whether each
        token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of
        the min was lower than the LHS.
    """

    ratio: torch.Tensor = torch.exp(policy_log_probs - old_log_probs)
    loss: torch.Tensor = -torch.min(
        ratio * advantages, torch.clip(ratio, (1 - cliprange), (1 + cliprange)) * advantages
    )

    metadata: dict[str, torch.Tensor] = {}

    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str | Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
        policy being trained.
        loss_type One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).
        advantages Required for "reinforce_with_baseline" and "grpo_clip"; shape
        (batch_size, 1).
        old_log_probs Required for "grpo_clip"; shape (batch_size, sequence_length).
        cliprange Required for "grpo_clip"; scalar ε used for clipping.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss (batch_size, sequence_length), per-token loss.
        metadata dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    """

    if loss_type == "grpo_clip":
        assert old_log_probs is not None, "old_log_probs required for grpo_clip loss"
        assert cliprange is not None, "cliprange required for grpo_clip loss"
        assert advantages is not None, "advantages required for grpo_clip loss"
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
        return loss, metadata

    elif loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards required for no_baseline loss"
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}

    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages required for reinforce_with_baseline loss"
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
