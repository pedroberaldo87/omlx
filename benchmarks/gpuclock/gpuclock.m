// gpuclock: per-command-buffer GPU timing for any Metal process (MLX included).
// Build:  clang -O2 -dynamiclib -fobjc-arc -framework Metal -framework Foundation gpuclock.m -o gpuclock.dylib
// Use:    ctypes.CDLL("gpuclock.dylib"); lib.gpuclock_arm(); ... lib.gpuclock_drain(buf, n)
#import <Metal/Metal.h>
#import <objc/runtime.h>
#import <os/lock.h>
#import <QuartzCore/QuartzCore.h>

#define CAP 200000
typedef struct { double kstart, kend, gstart, gend; } Rec;
static Rec g_recs[CAP];
static int g_n = 0;
static os_unfair_lock g_lock = OS_UNFAIR_LOCK_INIT;

static void record(id<MTLCommandBuffer> b) {
  Rec r = { b.kernelStartTime, b.kernelEndTime, b.GPUStartTime, b.GPUEndTime };
  os_unfair_lock_lock(&g_lock);
  if (g_n < CAP) g_recs[g_n++] = r;
  os_unfair_lock_unlock(&g_lock);
}

static IMP orig_unretained = NULL;
static IMP orig_plain = NULL;

static id hook_unretained(id self, SEL _cmd) {
  id cb = ((id (*)(id, SEL))orig_unretained)(self, _cmd);
  if (cb) [cb addCompletedHandler:^(id<MTLCommandBuffer> b) { record(b); }];
  return cb;
}
static id hook_plain(id self, SEL _cmd) {
  id cb = ((id (*)(id, SEL))orig_plain)(self, _cmd);
  if (cb) [cb addCompletedHandler:^(id<MTLCommandBuffer> b) { record(b); }];
  return cb;
}

int gpuclock_arm(void) {
  static int armed = 0;
  if (armed) return 0;
  id<MTLDevice> d = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> q = [d newCommandQueue];          // concrete class discovery, no hardcoded AGX name
  Class qc = object_getClass(q);
  Method m1 = class_getInstanceMethod(qc, @selector(commandBufferWithUnretainedReferences));
  Method m2 = class_getInstanceMethod(qc, @selector(commandBuffer));
  if (!m1 || !m2) return -1;
  orig_unretained = method_setImplementation(m1, (IMP)hook_unretained);
  orig_plain      = method_setImplementation(m2, (IMP)hook_plain);
  armed = 1;
  return 0;
}

int gpuclock_count(void) { return g_n; }
void gpuclock_reset(void) { os_unfair_lock_lock(&g_lock); g_n = 0; os_unfair_lock_unlock(&g_lock); }

// Copies up to max records (4 doubles each) into out; returns how many.
int gpuclock_drain(double *out, int max) {
  os_unfair_lock_lock(&g_lock);
  int n = g_n < max ? g_n : max;
  for (int i = 0; i < n; i++) {
    out[4*i+0] = g_recs[i].kstart; out[4*i+1] = g_recs[i].kend;
    out[4*i+2] = g_recs[i].gstart; out[4*i+3] = g_recs[i].gend;
  }
  g_n = 0;
  os_unfair_lock_unlock(&g_lock);
  return n;
}

// Same clock as GPUStartTime (mach_absolute seconds), for python-side cycle marks.
double gpuclock_now(void) { return CACurrentMediaTime(); }
