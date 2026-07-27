"""BPE tokenizer for Hearth, built on HuggingFace's `tokenizers` library
(fast, pure Rust, no dependency on transformers). Train it once on your
corpus, then load it everywhere - the encoder and every head share the
same vocab.
"""
from pathlib import Path
from typing import List

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors, decoders

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"]


def train_tokenizer(
    files: List[str],
    save_path: str,
    vocab_size: int = 32_000,
    min_frequency: int = 2,
):
    """Train a byte-level BPE tokenizer on a list of plain-text files
    (one training example / conversation turn per line works well) and
    save it to `save_path` (a single .json file).
    """
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train(files, trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(save_path)
    return tokenizer


class HearthTokenizer:
    """Thin wrapper exposing the encode/decode API the rest of Hearth uses,
    with sane padding/truncation defaults matching HearthConfig.max_seq_len."""

    def __init__(self, tokenizer_path: str, max_seq_len: int = 512):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_seq_len = max_seq_len
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.tokenizer.enable_padding(pad_id=self.pad_id, pad_token="<pad>", length=None)
        self.tokenizer.enable_truncation(max_length=max_seq_len)

    def encode(self, text: str):
        enc = self.tokenizer.encode(text)
        return enc.ids, enc.attention_mask

    def encode_batch(self, texts: List[str]):
        encs = self.tokenizer.encode_batch(texts)
        input_ids = [e.ids for e in encs]
        attention_mask = [e.attention_mask for e in encs]
        return input_ids, attention_mask

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the Hearth BPE tokenizer")
    parser.add_argument("--files", nargs="+", required=True, help="Plain text training file(s)")
    parser.add_argument("--out", default="hearth_ai/tokenizer/hearth_tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=32_000)
    args = parser.parse_args()

    train_tokenizer(args.files, args.out, vocab_size=args.vocab_size)
    print(f"Saved tokenizer to {args.out}")
