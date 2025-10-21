import torch
from smiles_bd.model import TransformerDenoiser
def test_model_forward_shape():
    vocab=16
    model=TransformerDenoiser(vocab_size=vocab, d_model=64, n_heads=4, n_layers=2, max_len=32, disable_nested_tensor=True)
    x=torch.randint(0, vocab, (2,32)); attn=torch.ones_like(x)
    y=model(x, attention_mask=attn); assert y.shape==(2,32,vocab)
