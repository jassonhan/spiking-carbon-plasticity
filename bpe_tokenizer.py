#!/usr/bin/env python3
"""
bpe_tokenizer.py —— 纯标准库 + numpy 的字符级 BPE 分词器（无外部依赖）

设计要点：
- 初始词表 = 语料中所有唯一字符（中文天然 char-level 起步）。
- 在语料「子集」上训练 merge 规则（统计相邻 pair 频次用 numpy bincount，向量化合并），
  学到的规则泛化到全量语料。
- 词表大小 = 基础字符数 + merge 次数。可精确控制在 20M 参数预算内。
- 对大文件编码：按 rank 顺序逐次向量化合并，得到 token id 数组（缓存为 .npy）。

用法：
  from bpe_tokenizer import BPETokenizer
  tok = BPETokenizer()
  tok.train(open("train.txt",encoding="utf-8").read()[:1_500_000], target_vocab=10600)
  tok.save("bpe_chinese.json")
  ids = tok.encode("我爱北京天安门")
  text = tok.decode(ids)
"""
import json
from pathlib import Path

import numpy as np


def _merge_array(arr: np.ndarray, a: int, b: int, new_id: int) -> np.ndarray:
    """把 arr 中所有相邻 (a,b) 合并为 new_id，向量化、非重叠处理。"""
    n = arr.shape[0]
    if n < 2:
        return arr
    m = (arr[:-1] == a) & (arr[1:] == b)
    if not m.any():
        return arr
    # 非重叠起点：当前位置是 merge 起点且前一个位置不是 merge 起点
    start = m & ~np.concatenate((np.array([False]), m[:-1]))
    starts = np.where(start)[0]
    drop = np.zeros(n, dtype=bool)
    drop[starts + 1] = True
    keep = ~drop
    new_arr = arr[keep].copy()
    cum = np.cumsum(keep.astype(np.int64)) - 1  # 每个保留元素在新数组中的下标
    new_arr[cum[starts]] = new_id
    return new_arr


class BPETokenizer:
    def __init__(self):
        self.char2id = {}
        self.id2char = {}
        self.id2sym = {}          # id -> 字符串（基础字符或合并后的子词）
        self.merges = []           # [(a, b), ...] 按 rank 顺序
        self.merge_rank = {}       # (a, b) -> rank
        self.unk_id = 0
        self.base_vocab = 0
        self.vocab_size = 0

    # ---------------- 训练 ----------------
    def train(self, text: str, target_vocab: int = 10600, num_merges: int = None, subsample: int = 1_500_000):
        if subsample and len(text) > subsample:
            text = text[:subsample]
        chars = sorted(set(text))
        self.base_vocab = len(chars)
        self.char2id = {c: i for i, c in enumerate(chars)}
        self.id2char = {i: c for i, c in enumerate(chars)}
        self.id2sym = {i: c for i, c in enumerate(chars)}
        self.unk_id = 0  # 落在未知字符时归到 id 0（基础字符里通常含常见符号）
        # 初始符号序列
        syms = np.array([self.char2id[c] for c in text], dtype=np.int64)
        V = self.base_vocab
        next_id = V
        if num_merges is None:
            num_merges = max(0, target_vocab - V)
        for _ in range(num_merges):
            if syms.shape[0] < 2:
                break
            # 统计相邻 pair 频次（稀疏：只统计实际出现的 pair，避免 V*V 桶的 bincount）
            pairs = syms[:-1] * V + syms[1:]
            uniq, cnts = np.unique(pairs, return_counts=True)
            if uniq.size == 0:
                break
            bi = int(np.argmax(cnts))
            if cnts[bi] <= 1:
                break
            best = int(uniq[bi])
            a = best // V
            b = best % V
            syms = _merge_array(syms, a, b, next_id)
            self.merges.append((int(a), int(b)))
            self.merge_rank[(int(a), int(b))] = len(self.merges) - 1
            self.id2sym[next_id] = self.id2sym[a] + self.id2sym[b]
            next_id += 1
            V += 1
        self.vocab_size = next_id
        print(f"[bpe] base={self.base_vocab} merges={len(self.merges)} vocab={self.vocab_size}")

    # ---------------- 编码（短文本，生成时用） ----------------
    def encode(self, text: str) -> list:
        ids = [self.char2id.get(c, self.unk_id) for c in text]
        while True:
            best_rank = None
            best_i = -1
            for i in range(len(ids) - 1):
                r = self.merge_rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_i == -1:
                break
            ids = ids[:best_i] + [self.vocab_size - len(self.merges) + best_rank] + ids[best_i + 2:]
            # 上面 new_id = base_vocab + rank
        return ids

    # ---------------- 编码（大文件，向量化） ----------------
    def _build_cp_lut(self):
        """构建 code-point -> id 的查找数组，避免逐字符 Python 字典查找（性能瓶颈）。"""
        if not self.char2id:
            maxcp = 0
        else:
            maxcp = max(ord(c) for c in self.char2id)
        # 末位多留一格专门放 unk，避免越界；clip 到 maxcp+1 即为未知字
        lut = np.full(maxcp + 2, self.unk_id, dtype=np.int64)
        for c, i in self.char2id.items():
            lut[ord(c)] = i
        self._cp_lut = lut
        self._cp_max = maxcp + 1

    def encode_array(self, text: str, verbose: bool = False) -> np.ndarray:
        """把整段文本向量化编码为 token id 数组（char->id 用查表，全向量化）。"""
        if not hasattr(self, "_cp_lut"):
            self._build_cp_lut()
        # utf-32-le：每个码位固定 4 字节，frombuffer 直接得到 code point
        cp = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32).astype(np.int64)
        cp = np.clip(cp, 0, self._cp_max)
        ids = self._cp_lut[cp]
        n = len(self.merges)
        for rank, (a, b) in enumerate(self.merges):
            new_id = self.base_vocab + rank
            ids = _merge_array(ids, a, b, new_id)
            if verbose and (rank + 1) % 200 == 0:
                print(f"  [encode] {rank+1}/{n}  tokens={ids.shape[0]:,}", flush=True)
        return ids

    # ---------------- 解码 ----------------
    def decode(self, ids) -> str:
        return "".join(self.id2sym.get(int(i), "") for i in ids)

    # ---------------- 序列化 ----------------
    def save(self, path: str):
        Path(path).write_text(
            json.dumps(
                {
                    "base_vocab": self.base_vocab,
                    "vocab_size": self.vocab_size,
                    "unk_id": self.unk_id,
                    "id2sym": {str(k): v for k, v in self.id2sym.items()},
                    "merges": self.merges,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[bpe] saved -> {path}")

    @classmethod
    def load(cls, path: str):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.base_vocab = d["base_vocab"]
        tok.vocab_size = d["vocab_size"]
        tok.unk_id = d["unk_id"]
        tok.id2sym = {int(k): v for k, v in d["id2sym"].items()}
        tok.merges = [tuple(m) for m in d["merges"]]
        tok.merge_rank = {tuple(m): i for i, m in enumerate(tok.merges)}
        # 重建 char2id/id2char
        tok.char2id = {}
        tok.id2char = {}
        for i in range(tok.base_vocab):
            c = tok.id2sym.get(i, "")
            tok.char2id[c] = i
            tok.id2char[i] = c
        return tok


if __name__ == "__main__":
    # 快速自检
    t = BPETokenizer()
    t.train("我爱北京天安门。我爱我的祖国。北京是中国的首都。", target_vocab=40)
    s = "我爱北京天安门"
    e = t.encode(s)
    print("encode:", e, "->", t.decode(e))
    assert t.decode(e) == s
    print("OK self-test passed")
