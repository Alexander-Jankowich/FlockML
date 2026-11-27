"""The sequence-model ops of bench_ops.ml ("sequence models" group) in
PyTorch, float64 on all threads, for a like-for-like comparison. Prints the
median microseconds per call.

    python3 bench/torch_ops.py
"""

import time, torch
torch.manual_seed(0)
d = torch.float64
x = torch.randn(32,64,128,dtype=d); w = torch.randn(128,128,dtype=d)
heads = torch.randn(32,64,4,32,dtype=d)
q = torch.randn(128,64,32,dtype=d); k = torch.randn(128,64,32,dtype=d)
att = torch.randn(128,64,64,dtype=d)
causal = torch.triu(torch.ones(64,64,dtype=torch.bool),1)
logits = torch.randn(2048,65,dtype=d); emb = torch.randn(65,128,dtype=d)
tok = torch.tensor([(i*31)%65 for i in range(2048)]).view(2048,1)
toks = tok.view(-1)
cases = [
 ("permute [32;64;4;32] -> heads", lambda: heads.permute(0,2,1,3).contiguous()),
 ("concat 2x [32;64;128] axis -1", lambda: torch.cat([x,x],-1)),
 ("slice [32;64;128] -> [:, :, 64:]", lambda: x[:,:,64:].contiguous()),
 ("matmul [32;64;128] x [128;128]", lambda: x @ w),
 ("bmm q k^T [128;64;32] -> 64x64", lambda: torch.bmm(q, k.transpose(1,2))),
 ("bmm att v [128;64;64] x [..;32]", lambda: torch.bmm(att, q)),
 ("masked_fill [128;64;64] causal", lambda: att.masked_fill(causal, -1e9)),
 ("logsumexp [2048;65] axis -1", lambda: torch.logsumexp(logits,-1)),
 ("gather [2048;65] one per row", lambda: torch.gather(logits,1,tok)),
 ("index_select 2048 rows of [65;128]", lambda: torch.index_select(emb,0,toks)),
]
print("PyTorch", torch.__version__, "threads", torch.get_num_threads())
for name, f in cases:
    for _ in range(20): f()
    ts=[]
    t_end=time.perf_counter()+0.3
    while time.perf_counter()<t_end or len(ts)<20:
        t0=time.perf_counter(); f(); ts.append(time.perf_counter()-t0)
    ts.sort(); print(f"{name}|{ts[len(ts)//2]*1e6:.1f}")
