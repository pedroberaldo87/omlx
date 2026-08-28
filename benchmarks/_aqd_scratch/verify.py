import time, random, statistics as st, mlx.core as mx
def one(f,n=30):
    mx.synchronize(); t=time.perf_counter()
    for _ in range(n): mx.eval(f())
    mx.synchronize(); return (time.perf_counter()-t)/n*1000
D,H=2560,2560
W={dt: mx.quantize(mx.random.normal((H,D)).astype(dt),group_size=64,bits=4) for dt in (mx.float16,mx.bfloat16)}
for M in (1,8,16,24,48,64):
    fs={}
    for dt in (mx.float16,mx.bfloat16):
        wq,s,b=W[dt]; x=mx.random.normal((M,D)).astype(dt); mx.eval(x,wq,s,b)
        fs[dt]=lambda x=x,wq=wq,s=s,b=b: mx.quantized_matmul(x,wq,s,b,transpose=True,group_size=64,bits=4)
        for _ in range(10): mx.eval(fs[dt]())
    smp={dt:[] for dt in fs}; names=list(fs)
    for _ in range(40):
        random.shuffle(names)
        for dt in names: smp[dt].append(one(fs[dt]))
    a,b_=st.median(smp[mx.float16]),st.median(smp[mx.bfloat16])
    print(f"qmm4 M={M:>2}: fp16={a:.4f} bf16={b_:.4f} ms  bf16/fp16={b_/a:.3f}x")
