import time, statistics as st, mlx.core as mx
def one(f,n=20):
    mx.synchronize(); t=time.perf_counter()
    for _ in range(n): mx.eval(f())
    mx.synchronize(); return (time.perf_counter()-t)/n*1000
a=mx.random.normal((2048,262144)).astype(mx.bfloat16); mx.eval(a)
f=lambda: mx.sum(a)
for _ in range(5): mx.eval(f())
t=st.median([one(f,10) for _ in range(9)]); print(f"stream sum 1.07GB bf16: {t:.3f} ms -> {a.nbytes/1e9/(t/1e3):.0f} GB/s"); del a
b=mx.random.normal((2048,262144)).astype(mx.float16); mx.eval(b)
g=lambda: mx.sum(b)
for _ in range(5): mx.eval(g())
t2=st.median([one(g,10) for _ in range(9)]); print(f"stream sum 1.07GB fp16: {t2:.3f} ms -> {b.nbytes/1e9/(t2/1e3):.0f} GB/s  bf16/fp16={t/t2:.3f}x"); del b
W=mx.random.normal((16384,8192)).astype(mx.bfloat16); wq,s,bi=mx.quantize(W,group_size=64,bits=4); del W
x=mx.random.normal((1,8192)).astype(mx.bfloat16); mx.eval(wq,s,bi,x)
h=lambda: mx.quantized_matmul(x,wq,s,bi,transpose=True,group_size=64,bits=4)
for _ in range(10): mx.eval(h())
t3=st.median([one(h,40) for _ in range(15)]); nb=wq.nbytes+s.nbytes+bi.nbytes
print(f"qmv4 16384x8192 ({nb/1e6:.0f}MB) M=1: {t3:.4f} ms -> {nb/1e9/(t3/1e3):.0f} GB/s")
