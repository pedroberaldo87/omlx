import time, random, statistics as st, mlx.core as mx
D=H=8192
def one(f,n=40):
    mx.synchronize(); t=time.perf_counter()
    for _ in range(n): mx.eval(f())
    mx.synchronize(); return (time.perf_counter()-t)/n*1000
cfgs={}
W = mx.random.normal((H,D)).astype(mx.bfloat16)
x = mx.random.normal((1,D)).astype(mx.bfloat16); mx.eval(W,x)
for name, kw in [("affine4_g64", dict(mode="affine",group_size=64,bits=4)),
                 ("affine4_g32", dict(mode="affine",group_size=32,bits=4)),
                 ("mxfp4",       dict(mode="mxfp4")),
                 ("nvfp4",       dict(mode="nvfp4"))]:
    try:
        r = mx.quantize(W, **kw)
        wq, s = r[0], r[1]; b = r[2] if len(r)>2 else None
        mx.eval(*[a for a in r if a is not None])
        bytes_ = wq.nbytes + s.nbytes + (b.nbytes if b is not None else 0)
        args = dict(kw); args.pop("mode",None)
        f = (lambda wq=wq,s=s,b=b,kw=kw: mx.quantized_matmul(x,wq,s,b,transpose=True,**{k:v for k,v in kw.items() if k!='mode'},mode=kw['mode']))
        f(); mx.eval(f())
        cfgs[name]=(f, bytes_, s.dtype, (b.dtype if b is not None else None))
        print(f"{name}: bytes={bytes_/1e6:.1f} MB ({bytes_*8/(H*D):.2f} bits/w) scales={s.dtype} biases={b.dtype if b is not None else None}")
    except Exception as e:
        print(f"{name}: UNSUPPORTED -> {type(e).__name__}: {str(e)[:150]}")
names=list(cfgs)
for f,_,_,_ in cfgs.values():
    for _ in range(10): mx.eval(f())
samples={n:[] for n in names}
for _ in range(50):
    random.shuffle(names)
    for n in names: samples[n].append(one(cfgs[n][0]))
base=st.median(samples.get("affine4_g64",[1]))
for n in cfgs:
    m=st.median(samples[n]); gb=cfgs[n][1]/1e9
    print(f"{n}: {m:.4f} ms  {gb/(m/1e3):.0f} GB/s  vs affine4_g64 = {m/base:.3f}x")
