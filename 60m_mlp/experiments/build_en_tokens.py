# -*- coding: utf-8 -*-
"""英文维基 → 中文 BPE token 流（供 Stage3 英语 T1 使用）。

- 读 E:\\english_corpus\\wikipedia_en\\*.parquet 前若干篇
- 只保留纯 ASCII 段落（去掉混入的非英文），长度筛选
- 用 bpe_chinese_wanjuan 编码（字符级，1.12 chars/token）
- 输出 int32 npy（E:\\english_corpus\\wiki_en_bpe.npy）+ 统计
"""
import os
import glob
import re
import sys
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bpe_tokenizer import BPETokenizer

OUT = r"E:\english_corpus\wiki_en_bpe.npy"
MAX_CHARS = 120_000_000          # 编码上限 1.2 亿字符 ≈ 1.07 亿 token
files = sorted(glob.glob(r"E:\english_corpus\wikipedia_en\*.parquet"))

tok = BPETokenizer.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "bpe_chinese_wanjuan.json"))

buf = []
total = 0
for fp in files:
    t = pq.read_table(fp, columns=["text"])
    texts = t.column("text").to_pylist()
    kept = 0
    for s in texts:
        if total >= MAX_CHARS:
            break
        # 纯 ASCII 判定（容忍换行/标点）
        non_ascii = sum(1 for ch in s[:800] if ord(ch) > 127)
        if non_ascii > len(s[:800]) * 0.01:
            continue
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) < 200:
            continue
        buf.append(s)
        total += len(s)
        kept += 1
    print("%s: +%d 篇 (累计 %.1f M chars)" % (os.path.basename(fp), kept, total / 1e6),
          flush=True)
    if total >= MAX_CHARS:
        break

print("编码 %d 篇 / %.1f M chars ..." % (len(buf), total / 1e6), flush=True)
ids = []
for i, s in enumerate(buf):
    ids.extend(tok.encode_array(s))
    if (i + 1) % 200 == 0:
        print("  已编码 %d/%d 篇, %d tokens" % (i + 1, len(buf), len(ids)), flush=True)

arr = np.asarray(ids, dtype=np.int32)
np.save(OUT, arr)
print("DONE: %s = %d tokens (%.1f chars/token)" % (OUT, len(arr), total / len(arr)))
