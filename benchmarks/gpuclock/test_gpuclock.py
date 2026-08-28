import ctypes, time
import mlx.core as mx

lib = ctypes.CDLL("/private/tmp/claude-501/-Users-pedroberaldo-omlx-qwen38-flash/a4f20682-db98-4b8a-9917-528167e7d904/scratchpad/gpuclock.dylib")
lib.gpuclock_now.restype = ctypes.c_double
assert lib.gpuclock_arm() == 0

# fake "decode cycle": 40 layers x (2 matmuls) on small tensors -> dispatch-bound like decode
W = [mx.random.normal((2048, 2048)).astype(mx.float16) for _ in range(8)]
x = mx.random.normal((1, 2048)).astype(mx.float16)
mx.eval(W, x)

BUF = (ctypes.c_double * (4 * 20000))()
for cycle in range(12):
    lib.gpuclock_drain(BUF, 20000)  # clear
    t0 = lib.gpuclock_now()
    h = x
    for _ in range(5):
        for w in W:
            h = mx.maximum(h @ w, 0.0)
    mx.eval(h)
    t1 = lib.gpuclock_now()
    time.sleep(0.002)  # let completion handlers land
    n = lib.gpuclock_drain(BUF, 20000)
    if n == 0:
        print(f"cycle {cycle}: NO RECORDS"); continue
    recs = [(BUF[4*i], BUF[4*i+1], BUF[4*i+2], BUF[4*i+3]) for i in range(n)]
    busy = sum(r[3] - r[2] for r in recs)
    span = max(r[3] for r in recs) - min(r[2] for r in recs)
    first_gap = min(r[2] for r in recs) - t0
    tail = t1 - max(r[3] for r in recs)
    if cycle >= 2:
        print(f"cycle {cycle:2d}: wall {(t1-t0)*1e3:7.3f} ms | cmdbufs {n:3d} | "
              f"gpu_busy {busy*1e3:7.3f} ms | gpu_span {span*1e3:7.3f} ms | "
              f"idle_in_span {(span-busy)*1e3:6.3f} ms | head {first_gap*1e3:6.3f} | tail {tail*1e3:6.3f}")
