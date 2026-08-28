import time, random, statistics as st, mlx.core as mx
D=H=4096
def mk_q4(dt):
    W=mx.random.normal((H,D)).astype(dt); wq,s,b=mx.quantize(W,group_size=64,bits=4)
    x=mx.random.normal((1,D)).astype(dt); mx.eval(x,wq,s,b)
    return lambda: mx.quantized_matmul(x,wq,s,b,transpose=True,group_size=64,bits=4)
def mk_dense(dt):
    x=mx.random.normal((1,D)).astype(dt); W=mx.random.normal((H,D)).astype(dt); mx.eval(x,W)
    return lambda: x@W.T
def one(f,n=50):
    mx.synchronize(); t=time.perf_counter()
    for _ in range(n): mx.eval(f())
    mx.synchronize(); return (time.perf_counter()-t)/n*1000

for label, mk in (("q4g64 matvec", mk_q4), ("dense gemv", mk_dense)):
    fs={dt:mk(dt) for dt in (mx.float16,mx.bfloat16)}
    for f in fs.values():
        for _ in range(20): mx.eval(f())
    samples={mx.float16:[],mx.bfloat16:[]}
    order=[mx.float16,mx.bfloat16]
    for _ in range(80):
        random.shuffle(order)
        for dt in order: samples[dt].append(one(fs[dt]))
    a=sorted(samples[mx.float16]); b=sorted(samples[mx.bfloat16])
    med=lambda v: st.median(v); p10=lambda v: v[len(v)//10]
    print(f"{label} 4096x4096 (80 paired, randomized order, n=50 each)")
    print(f"   fp16 median={med(a):.4f} p10={p10(a):.4f}")
    print(f"   bf16 median={med(b):.4f} p10={p10(b):.4f}")
    print(f"   bf16/fp16 median={med(b)/med(a):.3f}x   p10={p10(b)/p10(a):.3f}x")
