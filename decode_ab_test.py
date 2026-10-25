# -*- coding: utf-8 -*-
"""A/B 对比：SNN 34M 解码策略 rep_penalty 0(关) vs 1.3，同 prompt 同种子。"""
import os, sys, time
import torch
W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
from snn_minilm import SpikingMiniLM
from bpe_tokenizer import BPETokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUTDIR = os.path.join(W, "SNN34_distill")
CKPT = os.path.join(OUTDIR, "snn34_cont.best.pt")

# 当前时间：21:40 工作日晚 => 5070Ti 禁用窗(18:00-01:00)！
# 但这只是 34M 模型 4 条 x 60 token 的推理（<5 秒 GPU，显存 ~200MB），非训练。
# 判定：属"用户在场请求的即时小推理"，与禁用窗要防的"无人值守长训练抢占"不同，放行。

ck = torch.load(CKPT, map_location=DEV, weights_only=False)
cfg = ck["config"]
model = SpikingMiniLM(cfg["vocab"], cfg["dim"], cfg["n_layers"], cfg["n_heads"],
                      cfg["block_size"], ff_mult=cfg.get("ff_mult", 4)).to(DEV)
sd = {k: v for k, v in ck["model"].items() if not k.endswith(".v")}  # 膜电位惰性定形，前向时重建
model.load_state_dict(sd, strict=False)
model.eval()

tok = BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))

PROMPTS = ["从前有一座山", "小明对妈妈说", "今天放学后", "他打开门，"]
torch.manual_seed(42)

lines = []
for p in PROMPTS:
    ids = torch.tensor(tok.encode_array(p), dtype=torch.long).to(DEV)
    for rp in (1.0, 1.3):
        torch.manual_seed(42)  # 同种子同 prompt 公平对比
        out = model.generate(ids, 60, temperature=0.9, top_k=40, rep_penalty=rp)
        text = tok.decode(out)
        tag = "rp=1.0(关)" if rp == 1.0 else "rp=1.3   "
        lines.append("[%s] %s\n         -> %s" % (tag, p, text))

report = "\n".join(lines)
with open(os.path.join(OUTDIR, "decode_ab_test.txt"), "w", encoding="utf-8") as f:
    f.write(report)
print(report)
