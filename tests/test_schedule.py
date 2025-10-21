import torch
from smiles_bd.schedule import ClippedLinearSchedule

def test_schedule_range_and_weight():
    sch=ClippedLinearSchedule(0.2,0.7)
    r=sch.sample_mask_rate((1000,), device="cpu")
    assert (r.min()>=0.2-1e-6) and (r.max()<=0.7+1e-6)
    w=sch.loss_weight(r); assert torch.all(torch.isfinite(w))
