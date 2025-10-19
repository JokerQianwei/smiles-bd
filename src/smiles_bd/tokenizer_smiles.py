
import collections
import re
from typing import List, Dict, Optional

class RegexSmilesTokenizer:
    """
    Minimal SMILES tokenizer:
    - chemistry-aware regex (extended)
    - load vocab.txt (one token per line)
    - never auto-inject special tokens
    """
    SMI_REGEX_PATTERN = (
        r"(\[[^\[\]]{1,6}\]|Br?|Cl?|Si|Se|Na|Ca|Li|Mg|Al|Fe|K|N|O|S|P|F|I|B|C|H|"
        r"[cnops]|%[0-9]{2}|[0-9]|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$)"
    )

    def __init__(self, vocab_path: str):
        self.regex = re.compile(self.SMI_REGEX_PATTERN)
        self.vocab: Dict[str, int] = self._load_vocab(vocab_path)
        self.inv_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        # required tokens
        for tok in ["[PAD]", "[MASK]", "[EOS]"]:
            if tok not in self.vocab:
                raise ValueError(f"'{tok}' must exist in vocab.txt")
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.eos_token = "[EOS]"
        self.pad_token_id = self.vocab[self.pad_token]
        self.mask_token_id = self.vocab[self.mask_token]
        self.eos_token_id = self.vocab[self.eos_token]
        self.unk_token_id = self.vocab.get("[UNK]", self.eos_token_id)

    @staticmethod
    def _load_vocab(vocab_file: str) -> Dict[str, int]:
        vocab = collections.OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as r:
            for i, line in enumerate(r):
                tok = line.rstrip("\n")
                if tok == "": continue
                vocab[tok] = i
        return vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize(self, text: str) -> List[str]:
        # remove spaces to avoid splitting special tokens written with spaces
        text = text.replace(" ", "")
        return [t for t in self.regex.findall(text)]

    def encode(self, text: str, add_special_tokens: bool = False, max_length: Optional[int] = None,
               padding: bool = False, truncation: bool = True) -> List[int]:
        toks = self.tokenize(text)
        ids = [self.vocab.get(t, self.unk_token_id) for t in toks]
        if max_length is not None:
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            if padding and len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
        return ids

    def batch_encode(self, texts, max_length: Optional[int], padding: bool, truncation: bool):
        return [self.encode(t, max_length=max_length, padding=padding, truncation=truncation) for t in texts]

    def decode(self, ids: List[int]) -> str:
        parts = []
        for i in ids:
            if int(i) == self.pad_token_id:
                continue
            parts.append(self.inv_vocab.get(int(i), ""))
        return "".join(parts)

    def batch_decode(self, batch_ids):
        return [self.decode(x) for x in batch_ids]
