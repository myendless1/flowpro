from flowpro.training.mixer import batch_counts
import importlib.util


def test_paper_batch_ratios():
    assert batch_counts(100,1)=={"current":80,"sft":20}
    assert batch_counts(100,2)=={"current":70,"history":15,"sft":15}


def test_rpro_identical_pair_has_finite_gradient():
    if importlib.util.find_spec("torch") is None:
        return
    import torch
    from flowpro.training.rpro import rpro_loss
    current = torch.tensor([1.0, 2.0], requires_grad=True)
    reference = torch.tensor([1.5, 1.5])
    loss, metrics = rpro_loss(current, current, reference, reference)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(current.grad).all()
    assert set(metrics) == {"loss/pro", "loss/sft", "reward/winner", "reward/loser"}
