import re
from typing import List, Optional, Dict

class RegexSmilesTokenizer:
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
        self.pad_token_id  = self.vocab["[PAD]"]
        self.mask_token_id = self.vocab["[MASK]"]
        self.eos_token_id  = self.vocab["[EOS]"]
        self.sep_token_id  = self.vocab["[SEP]"]
        self.unk_token_id  = self.vocab["[UNK]"]
        self.vocab_size    = len(self.vocab)

    def tokenize(self, text: str) -> List[str]:
        return [t for t in self.regex.findall(text.replace(" ", ""))]

    def encode(self, text: str, add_special_tokens: bool=False,
               max_length: Optional[int]=None, padding: bool=False, truncation: bool=True) -> List[int]:
        ids = [self.vocab.get(t, self.unk_token_id) for t in self.tokenize(text)]
        if max_length is not None:
            if truncation and len(ids) > max_length: ids = ids[:max_length]
            if padding and len(ids) < max_length:    ids = ids + [self.pad_token_id]*(max_length-len(ids))
        return ids

    def batch_encode(self, texts, max_length: Optional[int], padding: bool, truncation: bool):
        return [self.encode(t, add_special_tokens=False, max_length=max_length, padding=padding, truncation=truncation) for t in texts]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.inv_vocab.get(int(i), "") for i in ids if int(i) != self.pad_token_id)

    def batch_decode(self, batch_ids):
        return [self.decode(x) for x in batch_ids]
