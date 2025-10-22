import re
from typing import List, Dict, Optional, Iterable, Any

class RegexSmilesTokenizer:
    """
    Lightweight SMILES regex tokenizer with a fixed vocabulary.
    Designed to be compatible with Hugging Face datasets.map (batched).
    """
    SMI_REGEX_PATTERN = (
        r"(\[[^\[\]]{1,6}\]|Br?|Cl?|Si|Se|Na|Ca|Li|Mg|Al|Fe|K|N|O|S|P|F|I|B|C|H|"
        r"[cnops]|%[0-9]{2}|[0-9]|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$)"
    )

    def __init__(self, vocab_path: str):
        self.regex = re.compile(self.SMI_REGEX_PATTERN)
        self.vocab: Dict[str, int] = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for i, tok in enumerate(t.strip() for t in f if t.strip()):
                self.vocab[tok] = i
        self.inv_vocab = {i: t for t, i in self.vocab.items()}
        # required special tokens in vocab.txt
        self.pad_token_id = self.vocab.get("[PAD]")
        self.mask_token_id = self.vocab.get("[MASK]")
        self.sep_token_id = self.vocab.get("[SEP]")
        self.unk_token_id = self.vocab.get("[UNK]")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize(self, s: str) -> List[str]:
        return [m.group(0) for m in self.regex.finditer(s)]

    def encode(self, s: str, max_length: Optional[int] = None,
               padding: bool = False, truncation: bool = True) -> List[int]:
        toks = self.tokenize(s)
        ids = [self.vocab.get(t, self.unk_token_id) for t in toks]
        if max_length is not None:
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            if padding and len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
        return ids

    def batch_encode(self, texts: Iterable[str], max_length: Optional[int],
                     padding: bool, truncation: bool) -> List[List[int]]:
        return [self.encode(t, max_length=max_length, padding=padding, truncation=truncation) for t in texts]

    def __call__(self, batch: Dict[str, Any], text_key: str = "text",
                 max_length: Optional[int] = None, padding: bool = True, truncation: bool = True) -> Dict[str, Any]:
        # Adapter for datasets.map(batched=True)
        texts = batch[text_key]
        ids_batch = self.batch_encode(texts, max_length=max_length, padding=padding, truncation=truncation)
        attn = [[0 if i == self.pad_token_id else 1 for i in ids] for ids in ids_batch]
        return {"input_ids": ids_batch, "attention_mask": attn}

    # decode helpers
    def decode(self, ids: List[int]) -> str:
        return "".join(self.inv_vocab.get(int(i), "") for i in ids if int(i) != self.pad_token_id)

    def batch_decode(self, batch_ids: Iterable[Iterable[int]]) -> List[str]:
        return [self.decode(x) for x in batch_ids]
