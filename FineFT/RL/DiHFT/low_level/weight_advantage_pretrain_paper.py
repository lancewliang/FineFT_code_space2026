"""Paper-faithful Stage I loss functions and training entry point."""

import copy

import torch
import torch.nn.functional as F

from RL.DiHFT.low_level import weight_advantage_pretrain as baseline


parser = copy.deepcopy(baseline.parser)
parser.add_argument(
    "--neighbor_size",
    type=int,
    default=1,
    help="fixed learner neighbor count from FineFT Algorithm 2",
)


def calculate_paper_supervisor_kl_loss(
    learner_action_values: torch.Tensor,
    expert_action_values: torch.Tensor,
    learner_weights: torch.Tensor,
) -> torch.Tensor:
    """Return the batch mean of the paper's weighted per-learner KL loss."""

    learner_log_probabilities = F.log_softmax(learner_action_values, dim=-1)
    expert_log_probabilities = F.log_softmax(expert_action_values, dim=-1)
    learner_probabilities = learner_log_probabilities.exp()
    per_learner_kl = (
        learner_probabilities
        * (learner_log_probabilities - expert_log_probabilities.unsqueeze(1))
    ).sum(dim=-1)
    return (per_learner_kl * learner_weights).sum(dim=1).mean()


def construct_paper_weight_matrix(
    etd_losses: torch.Tensor,
    neighbor_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct Algorithm 2 weights with a fixed index neighborhood."""

    if etd_losses.ndim != 3 or etd_losses.shape[1] != etd_losses.shape[2]:
        raise ValueError("etd_losses must have shape (batch_size, N, N)")
    if neighbor_size < 0:
        raise ValueError("neighbor_size must be non-negative")

    batch_size, ensemble_size, _ = etd_losses.shape
    device = etd_losses.device
    chosen_indices = torch.diagonal(etd_losses, dim1=1, dim2=2).argmin(dim=1)
    minimum_indices = (chosen_indices - neighbor_size).clamp(min=0)
    maximum_indices = (chosen_indices + neighbor_size).clamp(
        max=ensemble_size - 1
    )
    spans = maximum_indices - minimum_indices
    safe_spans = spans.clamp_min(1)

    indices = torch.arange(ensemble_size, device=device).expand(batch_size, -1)
    in_range = (indices >= minimum_indices.unsqueeze(1)) & (
        indices <= maximum_indices.unsqueeze(1)
    )
    diagonal_weights = (
        1
        - (indices - chosen_indices.unsqueeze(1)).abs().float()
        / safe_spans.unsqueeze(1)
    ) * in_range

    index_distances = (
        indices.unsqueeze(2) - indices.unsqueeze(1)
    ).abs().float()
    off_diagonal_decay = (
        1 - index_distances / safe_spans.view(batch_size, 1, 1)
    ).clamp_min(0)
    weight_matrix = torch.minimum(
        diagonal_weights.unsqueeze(2),
        diagonal_weights.unsqueeze(1),
    ) * off_diagonal_decay

    return diagonal_weights.detach(), weight_matrix.detach()


def calculate_paper_partial_loss(
    etd_losses: torch.Tensor,
    neighbor_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Algorithm 2 weights while preserving the baseline TD reduction."""

    learner_weights, weight_matrix = construct_paper_weight_matrix(
        etd_losses,
        neighbor_size,
    )
    batch_loss = (etd_losses * weight_matrix).sum(dim=1).mean(
        dim=1,
        keepdim=True,
    )
    return learner_weights, batch_loss.mean()


class PaperWeightedContextsDQN(baseline.Weighted_Contexts_DQN):
    """Stage I trainer using the three paper-formula ablations."""

    def __init__(self, args):
        super().__init__(args)
        if args.neighbor_size < 0:
            raise ValueError("neighbor_size must be non-negative")
        self.neighbor_size = args.neighbor_size

    def _prepare_network_inputs(self, states, info):
        batch_size = states.shape[0]
        flattened_states = states.reshape(batch_size, -1)
        previous_action = info["previous_action"].float().unsqueeze(1)
        available_action = info["avaliable_action"]
        hour_count_down = info["funding_count_down_hour"].float().unsqueeze(1)
        minute_count_down = info["funding_count_down_minute"].float().unsqueeze(1)
        time_input = torch.cat(
            [hour_count_down, minute_count_down],
            dim=1,
        ).to(self.device)
        trading_info = info["trading_info"].float().to(self.device)
        return (
            flattened_states,
            time_input,
            previous_action,
            available_action,
            trading_info,
        )

    def update(
        self,
        states: torch.Tensor,
        info: dict,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        info_: dict,
        dones: torch.Tensor,
    ):
        batch_size = states.shape[0]
        (
            states,
            time_input,
            previous_action,
            available_action,
            trading_info,
        ) = self._prepare_network_inputs(states, info)
        (
            next_states,
            next_time_input,
            next_previous_action,
            next_available_action,
            next_trading_info,
        ) = self._prepare_network_inputs(next_states, info_)

        current_sa_values = baseline.evaluate_quantile_at_action(
            self.eval_net(
                state=states,
                time=time_input,
                previous_action=previous_action,
                avaliable_action=available_action,
                trading_info=trading_info,
            ),
            actions,
        )
        assert current_sa_values.shape == (batch_size, self.N, 1)
        with torch.no_grad():
            next_q = self.target_net.get_best_q(
                state=next_states,
                time=next_time_input,
                previous_action=next_previous_action,
                avaliable_action=next_available_action,
                trading_info=next_trading_info,
            )
            next_sa_values = next_q.unsqueeze(1)
            assert next_sa_values.shape == (batch_size, 1, self.N)
            target_sa_values = rewards[..., None] + (
                (1.0 - dones[..., None]) * self.gamma * next_sa_values
            )
            assert target_sa_values.shape == (batch_size, 1, self.N)

        etd_losses = target_sa_values - current_sa_values
        assert etd_losses.shape == (batch_size, self.N, self.N)
        if self.if_use_hubber_loss:
            etd_losses = baseline.calculate_huber_loss(etd_losses)
        learner_weights, partial_td_error_loss = calculate_paper_partial_loss(
            etd_losses,
            self.neighbor_size,
        )

        predicted_action_values = self.eval_net(
            state=states,
            time=time_input,
            previous_action=previous_action,
            avaliable_action=available_action,
            trading_info=trading_info,
        )
        assert predicted_action_values.shape == (
            batch_size,
            self.N,
            self.N_ACTIONS,
        )
        expert_action_values = baseline.recalculate_q_demonstration(
            info["q_value"],
            info["avaliable_action"],
        )
        kl_divergence = calculate_paper_supervisor_kl_loss(
            predicted_action_values,
            expert_action_values,
            learner_weights,
        )
        loss = partial_td_error_loss + kl_divergence * self.ada
        baseline.update_params(
            self.optimizer,
            loss,
            self.eval_net,
            retain_graph=False,
            grad_cliping=self.grad_clip,
        )
        baseline.soft_copy_params(self.eval_net, self.target_net, self.tau)
        self.update_counter += 1
        return loss.item(), kl_divergence.item(), partial_td_error_loss.item()

    def update_pretrain(
        self,
        states: torch.Tensor,
        info: dict,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        info_: dict,
        dones: torch.Tensor,
    ):
        batch_size = states.shape[0]
        (
            states,
            time_input,
            previous_action,
            available_action,
            trading_info,
        ) = self._prepare_network_inputs(states, info)
        (
            next_states,
            next_time_input,
            next_previous_action,
            next_available_action,
            next_trading_info,
        ) = self._prepare_network_inputs(next_states, info_)

        current_sa_values = baseline.evaluate_quantile_at_action(
            self.eval_net(
                state=states,
                time=time_input,
                previous_action=previous_action,
                avaliable_action=available_action,
                trading_info=trading_info,
            ),
            actions,
        )
        assert current_sa_values.shape == (batch_size, self.N, 1)
        current_sa_values = current_sa_values.squeeze(-1)
        with torch.no_grad():
            next_q = self.target_net.get_best_q(
                state=next_states,
                time=next_time_input,
                previous_action=next_previous_action,
                avaliable_action=next_available_action,
                trading_info=next_trading_info,
            )
            next_sa_values = next_q.unsqueeze(1)
            assert next_sa_values.shape == (batch_size, 1, self.N)
            target_sa_values = rewards[..., None] + (
                (1.0 - dones[..., None]) * self.gamma * next_sa_values
            )
            target_sa_values = target_sa_values.permute(0, 2, 1)
            assert target_sa_values.shape == (batch_size, self.N, 1)
        target_sa_values = target_sa_values.squeeze(-1)

        td_loss = self.loss_func_pretrain(current_sa_values, target_sa_values)
        td_loss = td_loss.sum(dim=1).mean()
        learner_weights = torch.ones(
            batch_size,
            self.N,
            device=self.device,
        )
        predicted_action_values = self.eval_net(
            state=states,
            time=time_input,
            previous_action=previous_action,
            avaliable_action=available_action,
            trading_info=trading_info,
        )
        assert predicted_action_values.shape == (
            batch_size,
            self.N,
            self.N_ACTIONS,
        )
        expert_action_values = baseline.recalculate_q_demonstration(
            info["q_value"],
            info["avaliable_action"],
        )
        kl_divergence = calculate_paper_supervisor_kl_loss(
            predicted_action_values,
            expert_action_values,
            learner_weights,
        )
        loss = td_loss + kl_divergence * self.ada
        baseline.update_params(
            self.optimizer,
            loss,
            self.eval_net,
            retain_graph=False,
            grad_cliping=self.grad_clip,
        )
        baseline.soft_copy_params(self.eval_net, self.target_net, self.tau)
        self.update_counter += 1
        if torch.isnan(loss):
            baseline.log_loss_nan_diagnostics(
                logger=baseline.logger,
                numeric_values={
                    "loss": loss,
                    "KL_div": kl_divergence,
                    "td_loss": td_loss,
                    "states": states,
                    "next_states": next_states,
                    "actions": actions,
                    "rewards": rewards,
                    "dones": dones,
                    "time_input": time_input,
                    "next_time_input": next_time_input,
                    "previous_action": previous_action,
                    "next_previous_action": next_previous_action,
                    "avaliable_action": available_action,
                    "next_avaliable_action": next_available_action,
                    "current_sa_quantiles": current_sa_values,
                    "target_sa_quantiles": target_sa_values,
                    "predict_action_distrbution": predicted_action_values,
                    "q_value": expert_action_values,
                    "batch_weights": learner_weights,
                },
                info_values={"info": info, "info_": info_},
                trainer=self,
            )
            raise ValueError("loss is nan")
        return loss.item(), kl_divergence.item(), td_loss.item()


if __name__ == "__main__":
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    parsed_args = parser.parse_args()
    baseline.configure_logger(
        parsed_args.dataset_name,
        parsed_args.experiment_name,
    )
    baseline.logger.info(
        "start paper-formula Stage I | neighbor_size=%d",
        parsed_args.neighbor_size,
    )
    trainer = PaperWeightedContextsDQN(parsed_args)
    trainer.train()
