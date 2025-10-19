
import torch
from smiles_bd.tokenizer_smiles import RegexSmilesTokenizer
from smiles_bd.model import TransformerDenoiser
from smiles_bd.schedule import ClippedLinearSchedule
from smiles_bd.diffusion import MaskedDiffusion

def test_train_loop_smoke(tmp_path):
    vocab = tmp_path / "toy_vocab.txt"
    vocab.write_text("\n".join(["[PAD]","[MASK]","[EOS]","[UNK]","C","O","N","(",")","=","1"]), encoding="utf-8")
    tok = RegexSmilesTokenizer(str(vocab))
    # tiny synthetic batch
    L = 64
    ids = torch.tensor(tok.encode("CCO[EOS]" + "[PAD]"*(L-4), max_length=L, padding=True), dtype=torch.long).unsqueeze(0)
    attn = (ids != tok.pad_token_id).long()
    model = TransformerDenoiser(vocab_size=tok.vocab_size, d_model=64, n_heads=4, n_layers=2, max_len=L)
    schedule = ClippedLinearSchedule()
    diffuser = MaskedDiffusion(model, tok, schedule, pad_token_id=tok.pad_token_id, mask_token_id=tok.mask_token_id, eos_token_id=tok.eos_token_id, max_len=L)
    opt = torch.optim.AdamW(diffuser.parameters(), lr=1e-3)
    last = None
    for _ in range(10):
        out = diffuser.training_step({"input_ids": ids, "attention_mask": attn})
        out.loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        last = out.loss.item()
    assert last is not None and last < 10  # finite & reasonable
