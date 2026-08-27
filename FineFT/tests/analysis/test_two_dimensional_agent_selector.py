from pathlib import Path

import pandas as pd
import torch

from analysis.pick_agent.FineFT_two_dimensional_agent_selector import (
    assemble_and_save_ensemble,
)
from model.low_level import ensemble_Qnet


def test_assemble_and_save_ensemble_includes_flat_empty_model(
    tmp_path: Path,
) -> None:
    n_states = 3
    n_actions = 5
    hidden_nodes = 4
    time_info_dim = 2
    source = ensemble_Qnet(
        n_states,
        n_actions,
        hidden_nodes,
        time_info_dim,
        ensemble_number=1,
    )
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.25)
    source_path = tmp_path / "trained_model.pkl"
    torch.save(source.state_dict(), source_path)

    slots = pd.DataFrame(
        [
            {
                "slot_id": 0,
                "kind": "model",
                "model_path": str(source_path),
                "bin_index": 0,
            },
            {
                "slot_id": 1,
                "kind": "empty_model",
                "model_path": None,
                "bin_index": None,
            },
        ]
    )
    output_path = tmp_path / "model.pth"

    assemble_and_save_ensemble(
        slots,
        output_path,
        n_states=n_states,
        n_actions=n_actions,
        hidden_nodes=hidden_nodes,
        time_info_dim=time_info_dim,
    )

    assembled = ensemble_Qnet(
        n_states,
        n_actions,
        hidden_nodes,
        time_info_dim,
        ensemble_number=2,
    )
    assembled.load_state_dict(torch.load(output_path, weights_only=True))
    assert torch.equal(
        assembled.qnet_list[0].fc1.weight,
        source.qnet_list[0].fc1.weight,
    )

    batch_size = 3
    q_values = assembled(
        torch.randn(batch_size, n_states),
        torch.randn(batch_size, time_info_dim),
        torch.zeros(batch_size, 1),
        torch.ones(batch_size, n_actions),
        torch.randn(batch_size, 4),
    )
    assert q_values[:, 1].argmax(dim=1).tolist() == [n_actions // 2] * batch_size
