import logging
import sys
from pathlib import Path


FINEFT_ROOT = Path(__file__).resolve().parents[2]
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))

from RL.DiHFT.low_level.weight_advantage_pretrain import Weighted_Contexts_DQN, logger


class DummyLargeObject:
    pass


def test_log_internal_parameters_logs_values_and_summarizes_large_objects(caplog):
    trainer = Weighted_Contexts_DQN.__new__(Weighted_Contexts_DQN)
    trainer.dataset_name = "BTCUSDT"
    trainer.batch_size = 64
    trainer.position_list = [-8.0, 0, 8.0]
    trainer.eval_net = DummyLargeObject()

    with caplog.at_level(logging.INFO, logger=logger.name):
        trainer._log_internal_parameters("train_start")

    assert "Weighted_Contexts_DQN internal parameters | stage=train_start" in caplog.text
    assert "dataset_name=BTCUSDT" in caplog.text
    assert "batch_size=64" in caplog.text
    assert "position_list=[-8.0, 0, 8.0]" in caplog.text
    assert "eval_net=<DummyLargeObject>" in caplog.text
