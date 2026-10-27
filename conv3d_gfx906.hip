// FP16 NDHWC conv3d as an implicit GEMM on the gfx906 VALU pipe.
//
// gfx906 (MI50/MI60) has no WMMA/MFMA, so comfy_kitchen's WMMA conv cannot run
// here. This kernel keeps the upstream contract: A rows are gathered straight
// from an NDHWC-ordered x view (ConvPatchSrc, verbatim from
// comfy_kitchen/backends/hip/ops/conv3d_fp16.hip), B is the weight in its
// channels_last_3d (K, T, R, S, C) layout, the output is packed NZPQK, and the
// epilogue adds fp16 bias and an optional fp16 residual. Accumulation is fp32
// through v_dot2_f32_f16, matching upstream.
//
// Tile: 128x128 block, 4 waves of 64 lanes in a 2x2 grid, each wave 64x64,
// each thread an 8x8 fp32 register tile. A K tile is 32 halfs staged in LDS
// with an 80-byte row stride; lanes own the interleaved rows {i*8 + lane}, so
// one ds_read_b128 per row read covers 8 distinct bank quads conflict-free.
// (Rows 8 apart at any 16B-aligned stride alias onto the same bank quad -
// 8 * stride is always a multiple of the 128B bank wrap - so the strided
// ownership lm*8+i would 8-way-conflict and the interleaved map is required.)
//
// shortcut: grid is linear in (K, M) blocks - no grouped ordering for L2
// locality yet. Upgrade if profiling shows B-side L2 misses.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

namespace {

// One 16-byte chunk (8 fp16 channels) of the patch row under output position
// `row`. Verbatim from upstream conv3d_fp16.hip.
struct ConvPatchSrc {
    const uint8_t* x;
    int zpq, pq, q_out;
    int C;
    int64_t xw, xh, xd, xn;  // element strides
    int rs, s_taps;
    int sd, sh, sw;

    __forceinline__ __device__ uint4 chunk(int row, int kbyte) const {
        const int n = row / zpq;
        int rem = row - n * zpq;
        const int z = rem / pq;
        rem -= z * pq;
        const int p = rem / q_out;
        const int q = rem - p * q_out;

        const int e = kbyte / 2;
        const int tap = e / C;
        const int c0 = e - tap * C;
        const int t = tap / rs;
        const int r = (tap - t * rs) / s_taps;
        const int s = tap - t * rs - r * s_taps;

        const int64_t idx = n * xn + (z * sd + t) * xd + (p * sh + r) * xh + (q * sw + s) * xw + c0;
        return *reinterpret_cast<const uint4*>(x + 2 * idx);
    }
};

constexpr int kWave = 64;
constexpr int BM = 128, BN = 128;
constexpr int BKH = 32;                    // halfs of K per LDS tile
constexpr int BKB = BKH * 2;               // 64 bytes per tile row
constexpr int kThreads = 256;              // 4 waves
constexpr int kLdsStride = BKH + 8;        // 40 halfs = 80B: rows i*8+lm land on disjoint bank quads
constexpr int kChunksPerRow = BKB / 16;    // 4
constexpr int kChunks = BM * kChunksPerRow;
constexpr int kPerThread = kChunks / kThreads;  // 2
static_assert(BM == BN, "A and B staging share the chunk count");

__forceinline__ __device__ float dot2(float acc, __half2 a, __half2 b) {
    return __builtin_amdgcn_fdot2(a, b, acc, false);
}

// C[M, N] = A[M, Kg] @ B[N, Kg]^T, fp16 in, fp32 acc, fp16 out.
// A comes from the conv patch gather; B rows are contiguous kbytes each.
__global__ __launch_bounds__(kThreads) void conv3d_fp16_gfx906_kernel(
    const ConvPatchSrc src, const uint8_t* __restrict__ w,
    const __half* __restrict__ bias, const __half* __restrict__ resid,
    __half* __restrict__ out, int M, int N, int kbytes, int ldc) {

    __shared__ __align__(16) __half As[BM][kLdsStride];
    __shared__ __align__(16) __half Bs[BN][kLdsStride];

    const int tid = threadIdx.x;
    const int lane = tid & (kWave - 1);
    const int warp = tid / kWave;
    const int wm = warp >> 1;   // 2 waves along M
    const int wn = warp & 1;    // 2 waves along N
    const int lm = lane & 7;    // interleaved row ownership: thread covers rows i*8 + lm
    const int ln = lane >> 3;   // cols j*8 + ln

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    float acc[8][8] = {};
    uint4 areg[kPerThread], breg[kPerThread];

    // Issue this thread's share of a tile's global loads into registers.
    auto stage = [&](int kb0) {
        #pragma unroll
        for (int i = 0; i < kPerThread; ++i) {
            const int c = tid + i * kThreads;  // wave-coalesced 16B chunks
            const int grow = m0 + c / kChunksPerRow;
            const int gk = kb0 + (c % kChunksPerRow) * 16;
            areg[i] = (grow < M && gk < kbytes) ? src.chunk(grow, gk) : make_uint4(0, 0, 0, 0);
            const int growb = n0 + c / kChunksPerRow;
            breg[i] = (growb < N && gk < kbytes)
                          ? *reinterpret_cast<const uint4*>(
                                w + static_cast<int64_t>(growb) * kbytes + gk)
                          : make_uint4(0, 0, 0, 0);
        }
    };

    auto store_lds = [&]() {
        #pragma unroll
        for (int i = 0; i < kPerThread; ++i) {
            const int c = tid + i * kThreads;
            const int row = c / kChunksPerRow;
            const int lh = (c % kChunksPerRow) * 8;
            *reinterpret_cast<uint4*>(&As[row][lh]) = areg[i];
            *reinterpret_cast<uint4*>(&Bs[row][lh]) = breg[i];
        }
    };

    stage(0);
    store_lds();
    __syncthreads();

    for (int kb0 = 0; kb0 < kbytes; kb0 += BKB) {
        const bool has_next = kb0 + BKB < kbytes;
        if (has_next) {
            stage(kb0 + BKB);  // prefetch into registers while the math runs
        }

        #pragma unroll
        for (int ch = 0; ch < kChunksPerRow; ++ch) {
            __half2 av[8][4], bv[8][4];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint4 v = *reinterpret_cast<const uint4*>(
                    &As[wm * 64 + i * 8 + lm][ch * 8]);
                const __half2* h = reinterpret_cast<const __half2*>(&v);
                av[i][0] = h[0]; av[i][1] = h[1]; av[i][2] = h[2]; av[i][3] = h[3];
            }
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint4 v = *reinterpret_cast<const uint4*>(
                    &Bs[wn * 64 + j * 8 + ln][ch * 8]);
                const __half2* h = reinterpret_cast<const __half2*>(&v);
                bv[j][0] = h[0]; bv[j][1] = h[1]; bv[j][2] = h[2]; bv[j][3] = h[3];
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    #pragma unroll
                    for (int p = 0; p < 4; ++p)
                        acc[i][j] = dot2(acc[i][j], av[i][p], bv[j][p]);
        }

        if (has_next) {
            __syncthreads();
            store_lds();
            __syncthreads();
        }
    }

    // Epilogue: out = acc + bias[col] + resid[row*ldc + col], fp16 store.
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = m0 + wm * 64 + i * 8 + lm;
        if (r >= M) continue;
        __half* crow = out + static_cast<int64_t>(r) * ldc;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = n0 + wn * 64 + j * 8 + ln;
            if (col >= N) continue;
            float v = acc[i][j];
            if (bias) v += __half2float(bias[col]);
            if (resid) v += __half2float(resid[static_cast<int64_t>(r) * ldc + col]);
            crow[col] = __float2half(v);
        }
    }
}

}  // namespace

// Same contract as upstream launch_fp16_conv3d_kernel: false when the kernel
// declines the call, true once the launch is queued. xs_* are x's (w, h, d, n)
// element strides; xs_h == 0 means packed NDHWC. Returns fp16 into `out`,
// packed [N*Z*P*Q, K].
extern "C" bool launch_fp16_conv3d_gfx906(
    const void* x, const void* w, const void* bias, const void* resid, void* out, int N, int D,
    int H, int W, int C, int K, int T, int R, int S, int Z, int P, int Q, int sd, int sh, int sw,
    int64_t xs_w, int64_t xs_h, int64_t xs_d, int64_t xs_n, hipStream_t stream) {

    if (C % 8 != 0 || K % 8 != 0 || sd <= 0 || sh <= 0 || sw <= 0) {
        return false;
    }
    const int64_t m64 = static_cast<int64_t>(N) * Z * P * Q;
    const int64_t depth = static_cast<int64_t>(C) * T * R * S;
    if (m64 == 0 || K == 0) {
        return true;
    }
    if (m64 > 0x7fffffff || 2 * depth > 0x7fffffff) {
        return false;
    }
    const int M = static_cast<int>(m64);
    const int kbytes = static_cast<int>(2 * depth);

    const bool packed = xs_h == 0;
    const int64_t xw = packed ? C : xs_w;
    const int64_t xh = packed ? static_cast<int64_t>(W) * C : xs_h;
    const int64_t xd = packed ? static_cast<int64_t>(H) * W * C : xs_d;
    const int64_t xn = packed ? static_cast<int64_t>(D) * H * W * C : xs_n;
    const ConvPatchSrc src{static_cast<const uint8_t*>(x), Z * P * Q, P * Q, Q, C,
                           xw, xh, xd, xn, R * S, S, sd, sh, sw};

    dim3 grid((K + BN - 1) / BN, (M + BM - 1) / BM);
    conv3d_fp16_gfx906_kernel<<<grid, kThreads, 0, stream>>>(
        src, static_cast<const uint8_t*>(w), static_cast<const __half*>(bias),
        static_cast<const __half*>(resid), static_cast<__half*>(out), M, K, kbytes, K);
    return true;
}
