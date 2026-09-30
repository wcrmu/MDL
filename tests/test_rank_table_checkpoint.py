import copy

import pytest
import torch

from src.modules.rank_table import RankEmbeddingTable
from src.optim import ShardedRowWiseAdagrad
from src.train import _clip_grad_norm


def test_optimizer_and_clipping_follow_first_forward_storage_growth():
    table = RankEmbeddingTable(4)
    original = table.weight
    optimizer = ShardedRowWiseAdagrad([original], lr=.01, initial_accumulator_value=.1)
    output = table.lookup(0, torch.tensor([7, 11]))
    assert table.weight is not original
    assert optimizer.param_groups[0]["params"][0] is table.weight
    torch.testing.assert_close(optimizer.state[table.weight]["sum"],
                               torch.full((table.num_embeddings,), .1))
    output.sum().backward()
    _clip_grad_norm([original], .25)
    assert table.weight.grad.coalesce().values().norm().item() == pytest.approx(.25, abs=1e-6)
    before = table.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(table.weight, before)


def test_grown_rank_table_restore_and_optimizer_resume():
    table = RankEmbeddingTable(4)
    table.insert_ids(torch.tensor([7, 11, -10]))
    optimizer = ShardedRowWiseAdagrad([table.weight], lr=.01)
    table.lookup(0, torch.tensor([7, 11])).sum().backward()
    optimizer.step()
    saved = copy.deepcopy(table.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    restored = RankEmbeddingTable(4)
    parameter_id = id(restored.weight)
    restored.load_state_dict(saved)
    assert id(restored.weight) == parameter_id
    assert restored.get_extra_state()["current_step"] == table.get_extra_state()["current_step"]
    assert restored._key_to_row == table._key_to_row
    torch.testing.assert_close(restored.weight, table.weight)
    resumed_optimizer = ShardedRowWiseAdagrad([restored.weight], lr=.01)
    resumed_optimizer.load_state_dict(optimizer_state)
    for current, opt in ((table, optimizer), (restored, resumed_optimizer)):
        opt.zero_grad(set_to_none=True)
        current.lookup(0, torch.tensor([7, 11])).sum().backward()
        opt.step()
    torch.testing.assert_close(restored.weight, table.weight)


def test_rank_table_rejects_inconsistent_mapping():
    table = RankEmbeddingTable(4)
    table.insert_ids(torch.tensor([7]))
    saved = copy.deepcopy(table.state_dict())
    saved["_extra_state"]["live_rows"] = 5
    with pytest.raises(ValueError, match="live_rows"):
        RankEmbeddingTable(4).load_state_dict(saved)
