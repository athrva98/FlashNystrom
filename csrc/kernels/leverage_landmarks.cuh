/******************************************************************************
 * Leverage-seeded Voronoi-mean landmarks for Nystrom attention.  sm_80+.
 *
 * Pipeline (per tensor X in {Q, K}, per (batch*head)):
 *   1. gram:      G = X^T X                            (fp32 accumulate)
 *   2. prep:      lam = tr(G)/m;  M = (G + lam I)^-1   (Cholesky, in SMEM)
 *                 floor = ALPHA * d_eff / N,  d_eff = tr(M G)
 *   3. score:     l_i = x_i^T M x_i;  g_i = log(l_i + floor) + Gumbel_i
 *                 (Philox4x32-10 counter RNG -> deterministic per seed)
 *   4. top-m:     two-stage parallel top-m over g  ->  m seed indices
 *                 (== sampling m seeds w/o replacement  proportional to  l_i + floor;
 *                  Plackett-Luce via the Gumbel-max trick)
 *   5. assign:    Voronoi partition of rows by nearest seed (euclidean),
 *                 accumulate per-cell sums (SMEM-staged atomics)
 *   6. finalize:  X_tilde[c] = mean(cell c) * scale   (empty cell -> seed row)
 *
 * Why this algorithm: validated against exact ridge-leverage-score ground
 * truth at N=4096 (see validate.py / phase2.py). Landmarks that are MEANS
 * dominate landmarks that are selected rows end-to-end (selected rows destroy
 * the conditioning of pinv(softmax(Qt Kt^T))); leverage-seeded Voronoi means
 * beat plain segment means by up to ~2x end-to-end on clustered data and tie
 * elsewhere. RLS of the unnormalized exp kernel and random-feature leverage
 * were both tested and REJECTED by that harness.
 *
 * All device math cross-checked bit-level on CPU (cpu_check.cpp):
 * Cholesky-inverse vs Gauss-Jordan, Philox KAT vector, Gumbel top-m vs
 * sequential Plackett-Luce, two-stage top-m vs partial_sort.
 *
 * Integration fixes over the original authoring (compile-untested) draft:
 *   F1. prep/score dynamic SMEM > 48 KB now opts in via cudaFuncSetAttribute
 *       (D=128), with a clear error if it exceeds the device limit.
 *   F2. assign: the x-broadcast __shfl_sync is hoisted out of `if (c < m)` so
 *       every lane participates (was UB for m not a multiple of 32).
 *   F3. caller-facing lm_workspace_bytes(BH,N,D,m) that derives topm_blocks
 *       from the device instead of taking it as an un-computable argument.
 ******************************************************************************/
#pragma once

#include <math_constants.h>   // CUDART_INF_F (float -inf; MSVC's INFINITY is double)
#include <cstdint>            // uint32_t/uint64_t/uintptr_t (gcc: not transitive)
#include <cstddef>            // size_t
#include "utils.h"

namespace flash_nystrom {

// ------------------------------------------------------------------ constants

constexpr float LM_ALPHA       = 0.05f;  // uniform mixing floor (validated value)
constexpr int   LM_BLK         = 256;    // block size for gram/score/topm/assign
constexpr int   LM_TOPM_MAX    = 256;    // max landmarks supported by top-m path
constexpr int   LM_SORT_P2     = 512;    // bitonic buffer: next_pow2(LM_TOPM_MAX + LM_BLK)
constexpr int   LM_GRAM_TILE   = 32;     // rows staged per gram iteration

// sm_80 features used: opt-in dynamic SMEM > 48KB (assign kernel keeps seeds
// AND accumulators resident), fast global/shared fp32 atomics, full-rate
// __shfl_sync broadcast pipelines. cp.async staging in the gram kernel is a
// possible further ~10-20% win but is deliberately left out: this file ships
// compile-untested and a silent async-copy bug would corrupt G undetectably.

// ------------------------------------------------------------------- philox

// Philox4x32-10 (Salmon et al. 2011). Stateless: output is a pure function of
// (counter, key), so scores are reproducible for a fixed user seed and there is
// no RNG state to allocate. KAT-checked in cpu_check.cpp.
struct Philox4 { uint32_t v[4]; };

__device__ __forceinline__ Philox4 philox4x32_10(Philox4 c, uint32_t k0, uint32_t k1) {
    const uint32_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
    const uint32_t W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
    #pragma unroll
    for (int r = 0; r < 10; r++) {
        uint32_t hi0 = __umulhi(M0, c.v[0]), lo0 = M0 * c.v[0];
        uint32_t hi1 = __umulhi(M1, c.v[2]), lo1 = M1 * c.v[2];
        Philox4 n;
        n.v[0] = hi1 ^ c.v[1] ^ k0;  n.v[1] = lo1;
        n.v[2] = hi0 ^ c.v[3] ^ k1;  n.v[3] = lo0;
        c = n;
        k0 += W0; k1 += W1;
    }
    return c;
}
__device__ __forceinline__ float philox_u01(uint32_t x) {
    // (x + 0.5) * 2^-32: strictly inside (0,1), so log(-log(u)) is finite.
    return ((float)x + 0.5f) * 2.3283064365386963e-10f;
}

// ============================================================ 1. gram kernel
//
// G[bh] += tile^T tile over all row tiles. Register-tiled SYRK-style:
// LM_BLK = 256 threads as a 16x16 grid; thread (ti,tj) owns the
// (D/16)x(D/16) sub-block G[ti*RD.., tj*RD..], accumulated in registers over
// SMEM-staged row tiles (cp.async double buffer), atomically added to global
// G at the end. Works for D=64 (RD=4, 16 regs) and D=128 (RD=8, 64 regs).
//
// Grid: (BH, ceil(N / rows_per_block));  Block: LM_BLK.
// SMEM: 2 * LM_GRAM_TILE * D floats.

template <typename scalar_t, int D>
__global__ void lm_gram_kernel(
    const scalar_t* __restrict__ x,   // (BH, N, D)
    float* __restrict__ g,            // (BH, D, D), pre-zeroed
    int N, int rows_per_block
) {
    constexpr int RD = D / 16;                    // per-thread tile edge
    static_assert(D % 16 == 0, "D must be a multiple of 16");
    const int bh   = blockIdx.x;
    const int ti   = threadIdx.x / 16;            // 0..15
    const int tj   = threadIdx.x % 16;            // 0..15
    const int row0 = blockIdx.y * rows_per_block;
    const int row1 = min(row0 + rows_per_block, N);
    if (row0 >= N) return;

    const scalar_t* xb = x + static_cast<size_t>(bh) * N * D;

    extern __shared__ float smem[];               // [2][LM_GRAM_TILE][D]
    float* buf[2] = { smem, smem + LM_GRAM_TILE * D };

    float acc[RD][RD];
    #pragma unroll
    for (int a = 0; a < RD; a++)
        #pragma unroll
        for (int b = 0; b < RD; b++) acc[a][b] = 0.0f;

    // stage a LM_GRAM_TILE x D tile: LM_BLK threads, each moves
    // (LM_GRAM_TILE*D)/LM_BLK elements. scalar_t may be fp16/bf16 -> convert.
    auto stage = [&](int s, int r0) {
        const int elems = LM_GRAM_TILE * D;
        for (int e = threadIdx.x; e < elems; e += LM_BLK) {
            const int r = r0 + e / D, c = e % D;
            buf[s][e] = (r < row1) ? to_float(xb[static_cast<size_t>(r) * D + c]) : 0.0f;
        }
    };

    int nt = 0;
    stage(0, row0);
    __syncthreads();
    for (int r0 = row0; r0 < row1; r0 += LM_GRAM_TILE, nt ^= 1) {
        if (r0 + LM_GRAM_TILE < row1) stage(nt ^ 1, r0 + LM_GRAM_TILE);
        const float* t = buf[nt];
        const int rows = min(LM_GRAM_TILE, row1 - r0);
        for (int r = 0; r < rows; r++) {
            float ai[RD], aj[RD];
            #pragma unroll
            for (int a = 0; a < RD; a++) ai[a] = t[r * D + ti * RD + a];
            #pragma unroll
            for (int b = 0; b < RD; b++) aj[b] = t[r * D + tj * RD + b];
            #pragma unroll
            for (int a = 0; a < RD; a++)
                #pragma unroll
                for (int b = 0; b < RD; b++) acc[a][b] = fmaf(ai[a], aj[b], acc[a][b]);
        }
        __syncthreads();
    }

    float* gb = g + static_cast<size_t>(bh) * D * D;
    #pragma unroll
    for (int a = 0; a < RD; a++)
        #pragma unroll
        for (int b = 0; b < RD; b++)
            atomicAdd(&gb[(ti * RD + a) * D + (tj * RD + b)], acc[a][b]);
}

// ============================================================ 2. prep kernel
//
// Per (b,h), one block of D threads, everything in SMEM:
//   lam   = tr(G) / m
//   A     = G + lam I;  A = chol(A) = L (lower);  Linv = L^-1;  M = L^-T L^-1
//   d_eff = tr(M G) = sum_i l_i   (identity checked in cpu_check.cpp)
//   floor = LM_ALPHA * d_eff / N       (Gumbel mixing floor, per bh)
// Writes M (D*D) and floor (1) to global. Loop order matches the CPU-verified
// reference exactly (cholesky_inverse in cpu_check.cpp), parallelized:
//   - factorization: column sweep, threads parallel over rows i >= j
//   - inversion: thread j performs the forward substitution for column j
//   - M: threads sweep (i,j) pairs.
//
// Grid: (BH);  Block: D;  SMEM: (2*D*D + D) floats.

template <int D>
__global__ void lm_prep_kernel(
    const float* __restrict__ g,      // (BH, D, D)
    float* __restrict__ m_out,        // (BH, D, D)
    float* __restrict__ floor_out,    // (BH)
    int N, int m_landmarks
) {
    const int bh = blockIdx.x;
    const int t  = threadIdx.x;
    const float* gb = g + static_cast<size_t>(bh) * D * D;

    extern __shared__ float sm[];
    float* A  = sm;                   // D*D: G+lam I -> L -> Linv (in place)
    float* M  = sm + D * D;           // D*D
    float* rd = sm + 2 * D * D;       // D reduction scratch

    // lam = tr(G)/m
    rd[t] = gb[t * D + t];
    __syncthreads();
    if (t == 0) {
        float tr = 0.0f;
        for (int i = 0; i < D; i++) tr += rd[i];
        rd[0] = tr / (float)m_landmarks;
    }
    __syncthreads();
    const float lam = rd[0];

    for (int i = 0; i < D; i++) A[i * D + t] = gb[i * D + t] + (i == t ? lam : 0.0f);
    __syncthreads();

    // Cholesky, column sweep (left-looking; reference loop order)
    for (int j = 0; j < D; j++) {
        if (t >= j) {                                    // t plays the row index i
            float s = A[t * D + j];
            for (int p = 0; p < j; p++) s = fmaf(-A[t * D + p], A[j * D + p], s);
            A[t * D + j] = s;                            // pre-division value
        }
        __syncthreads();
        if (t == j) A[j * D + j] = sqrtf(A[j * D + j]);
        __syncthreads();
        if (t > j) A[t * D + j] /= A[j * D + j];
        __syncthreads();
    }
    // zero the strict upper triangle so Linv writes are clean
    for (int i = 0; i < D; i++) if (t > i) A[i * D + t] = 0.0f;
    __syncthreads();

    // Invert L: thread j owns column j of X = L^-1 (forward substitution).
    // RACE NOTE: the sequential reference reads original L entries of columns
    // p > j, which other threads overwrite when columns run concurrently. So
    // we snapshot L into the (otherwise still unused) M buffer and read L
    // exclusively from there; A receives X. Column j writes only column j of
    // A and reads only column j of A (rows p < i, already final) -> race-free.
    for (int e = t; e < D * D; e += D) M[e] = A[e];
    __syncthreads();
    {
        const int j = t;
        A[j * D + j] = 1.0f / M[j * D + j];              // X[j][j]
        for (int i = j + 1; i < D; i++) {
            float s = 0.0f;
            for (int p = j; p < i; p++) s = fmaf(M[i * D + p], A[p * D + j], s);
            A[i * D + j] = -s / M[i * D + i];
        }
        __syncthreads();
    }

    // M = Linv^T Linv (symmetric), then d_eff = sum M .* G  (G symmetric)
    for (int e = t; e < D * D; e += D) {
        const int i = e / D, j = e % D;
        if (j <= i) {
            float s = 0.0f;
            for (int p = i; p < D; p++) s = fmaf(A[p * D + i], A[p * D + j], s);
            M[i * D + j] = s; M[j * D + i] = s;
        }
    }
    __syncthreads();

    float part = 0.0f;
    for (int e = t; e < D * D; e += D) part = fmaf(M[e], gb[e], part);
    rd[t] = part;
    __syncthreads();
    if (t == 0) {
        float deff = 0.0f;
        for (int i = 0; i < D; i++) deff += rd[i];
        floor_out[bh] = LM_ALPHA * deff / (float)N;      // == ALPHA * mean(l_i)
    }
    float* mo = m_out + static_cast<size_t>(bh) * D * D;
    for (int e = t; e < D * D; e += D) mo[e] = M[e];
}

// =========================================================== 3. score kernel
//
// g_i = log(l_i + floor) + Gumbel,  l_i = x_i^T M x_i.
// One warp per row: lane l holds x[l + 32k] (K = D/32 elements). y_a = sum_b
// M[a][b] x_b via shuffle-broadcast of x_b (M staged in SMEM), each lane owns
// K outputs; l = sum_a y_a x_a warp-reduced. ~D^2/32 FMA per lane per row.
//
// Grid: (BH, ceil(N / rows_per_block));  Block: LM_BLK (8 warps);
// SMEM: D*D floats.

template <typename scalar_t, int D>
__global__ void lm_score_kernel(
    const scalar_t* __restrict__ x,   // (BH, N, D)
    const float* __restrict__ m_in,   // (BH, D, D)
    const float* __restrict__ floor_in,
    float* __restrict__ gscore,       // (BH, N)
    int N, int rows_per_block, uint32_t seed_lo, uint32_t seed_hi
) {
    constexpr int K = D / 32;                     // elems per lane
    static_assert(D % 32 == 0, "D must be a multiple of 32");
    const int bh   = blockIdx.x;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int nwarp = LM_BLK / 32;

    extern __shared__ float sM[];                 // D*D
    {
        const float* mi = m_in + static_cast<size_t>(bh) * D * D;
        for (int e = threadIdx.x; e < D * D; e += LM_BLK) sM[e] = mi[e];
    }
    __syncthreads();

    const float flr = floor_in[bh];
    const scalar_t* xb = x + static_cast<size_t>(bh) * N * D;
    float* gb = gscore + static_cast<size_t>(bh) * N;

    const int row0 = blockIdx.y * rows_per_block;
    const int row1 = min(row0 + rows_per_block, N);

    for (int r = row0 + warp; r < row1; r += nwarp) {
        float xr[K], y[K];
        #pragma unroll
        for (int k = 0; k < K; k++) {
            xr[k] = to_float(xb[static_cast<size_t>(r) * D + k * 32 + lane]);
            y[k]  = 0.0f;
        }
        #pragma unroll
        for (int kb = 0; kb < K; kb++)
            for (int l = 0; l < 32; l++) {
                const float xbv = __shfl_sync(0xffffffffu, xr[kb], l);
                const int b = kb * 32 + l;
                #pragma unroll
                for (int ka = 0; ka < K; ka++)
                    y[ka] = fmaf(sM[(ka * 32 + lane) * D + b], xbv, y[ka]);
            }
        float ell = 0.0f;
        #pragma unroll
        for (int k = 0; k < K; k++) ell = fmaf(y[k], xr[k], ell);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ell += __shfl_down_sync(0xffffffffu, ell, o);

        if (lane == 0) {
            Philox4 c{{ (uint32_t)r, (uint32_t)bh, seed_lo, seed_hi }};
            Philox4 o = philox4x32_10(c, 0xCAFEF00Du, 0xC0FFEE11u);
            const float u = philox_u01(o.v[0]);
            const float gum = -__logf(-__logf(u));
            gb[r] = __logf(fmaxf(ell, 0.0f) + flr) + gum;
        }
    }
}

// =========================================================== 4. top-m kernels
//
// Two-stage exact top-m (verified vs partial_sort in cpu_check.cpp):
//  stage A: each block scans a strided slice of gscore keeping a running
//           top-m in SMEM (bitonic sort of [current top-m | new chunk]);
//           writes its m candidates (value, index) to workspace.
//  stage B: one block runs the same routine over all A-candidates.
// m <= LM_TOPM_MAX. Padding uses -INF.

__device__ __forceinline__ void lm_bitonic_desc(float* v, int* ix, int P2) {
    // blockDim.x == LM_BLK threads sort P2 (pow2) elements descending.
    for (int k = 2; k <= P2; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int e = threadIdx.x; e < P2; e += LM_BLK) {
                const int p = e ^ j;
                if (p > e) {
                    const bool up = ((e & k) == 0);        // descending overall
                    if (up ? (v[e] < v[p]) : (v[e] > v[p])) {
                        float tv = v[e]; v[e] = v[p]; v[p] = tv;
                        int   ti = ix[e]; ix[e] = ix[p]; ix[p] = ti;
                    }
                }
            }
            __syncthreads();
        }
}

static __global__ void lm_topm_stageA_kernel(
    const float* __restrict__ gscore, // (BH, N)
    float* __restrict__ cand_v,       // (BH, nblocks, m)
    int*   __restrict__ cand_i,       // (BH, nblocks, m)
    int N, int m
) {
    const int bh = blockIdx.x, blk = blockIdx.y, nblk = gridDim.y;
    const float* gb = gscore + static_cast<size_t>(bh) * N;

    __shared__ float sv[LM_SORT_P2];
    __shared__ int   si[LM_SORT_P2];
    for (int e = threadIdx.x; e < LM_SORT_P2; e += LM_BLK) { sv[e] = -CUDART_INF_F; si[e] = -1; }
    __syncthreads();

    // chunk-strided slice: block blk scans chunks blk, blk+nblk, ... of LM_BLK
    // contiguous rows each (coalesced; any partition gives exact top-m)
    for (int base = blk * LM_BLK; base < N; base += nblk * LM_BLK) {
        const int i = base + threadIdx.x;                  // contiguous chunk
        sv[m + threadIdx.x] = (i < N) ? gb[i] : -CUDART_INF_F;
        si[m + threadIdx.x] = (i < N) ? i : -1;
        __syncthreads();
        lm_bitonic_desc(sv, si, LM_SORT_P2);               // top-m survives in [0,m)
        // clear the chunk zone for next round
        for (int e = m + threadIdx.x; e < LM_SORT_P2; e += LM_BLK) { sv[e] = -CUDART_INF_F; si[e] = -1; }
        __syncthreads();
    }
    float* cv = cand_v + (static_cast<size_t>(bh) * nblk + blk) * m;
    int*   ci = cand_i + (static_cast<size_t>(bh) * nblk + blk) * m;
    for (int e = threadIdx.x; e < m; e += LM_BLK) { cv[e] = sv[e]; ci[e] = si[e]; }
}

static __global__ void lm_topm_stageB_kernel(
    const float* __restrict__ cand_v, // (BH, nblocks, m)
    const int*   __restrict__ cand_i,
    int* __restrict__ seeds,          // (BH, m)
    int nblocks, int m
) {
    const int bh = blockIdx.x;
    const float* cv = cand_v + static_cast<size_t>(bh) * nblocks * m;
    const int*   ci = cand_i + static_cast<size_t>(bh) * nblocks * m;
    const int total = nblocks * m;

    __shared__ float sv[LM_SORT_P2];
    __shared__ int   si[LM_SORT_P2];
    for (int e = threadIdx.x; e < LM_SORT_P2; e += LM_BLK) { sv[e] = -CUDART_INF_F; si[e] = -1; }
    __syncthreads();

    for (int base = 0; base < total; base += LM_BLK) {
        const int i = base + threadIdx.x;
        sv[m + threadIdx.x] = (i < total) ? cv[i] : -CUDART_INF_F;
        si[m + threadIdx.x] = (i < total) ? ci[i] : -1;
        __syncthreads();
        lm_bitonic_desc(sv, si, LM_SORT_P2);
        for (int e = m + threadIdx.x; e < LM_SORT_P2; e += LM_BLK) { sv[e] = -CUDART_INF_F; si[e] = -1; }
        __syncthreads();
    }
    int* sb = seeds + static_cast<size_t>(bh) * m;
    for (int e = threadIdx.x; e < m; e += LM_BLK) sb[e] = si[e];
}

// ===================================================== 5a. seed gather kernel
//
// x_tilde[c] = x[seeds[c]] (unscaled; empty-cell fallback + final scaling in
// finalize). Also writes h[c] = 0.5*||seed_c||^2 for the assignment argmax.
// Grid: (BH, m); Block: D threads (D <= 1024).

template <typename scalar_t, int D>
__global__ void lm_seed_gather_kernel(
    const scalar_t* __restrict__ x,   // (BH, N, D)
    const int* __restrict__ seeds,    // (BH, m)
    float* __restrict__ seed_rows,    // (BH, m, D)  fp32 working copy
    float* __restrict__ half_norms,   // (BH, m)
    int N, int m
) {
    const int bh = blockIdx.x, c = blockIdx.y, t = threadIdx.x;
    const int r = seeds[static_cast<size_t>(bh) * m + c];
    const float v = to_float(x[(static_cast<size_t>(bh) * N + r) * D + t]);
    float* out = seed_rows + (static_cast<size_t>(bh) * m + c) * D;
    out[t] = v;

    __shared__ float rd[D];
    rd[t] = v * v;
    __syncthreads();
    // tree-reduce
    for (int o = D / 2; o > 0; o >>= 1) {
        if (t < o) rd[t] += rd[t + o];
        __syncthreads();
    }
    if (t == 0) half_norms[static_cast<size_t>(bh) * m + c] = 0.5f * rd[0];
}

// ======================================================= 5b. assign+accumulate
//
// For each processed row: c* = argmax_c ( x_i . s_c - 0.5||s_c||^2 )
// (== euclidean-nearest seed), then accumulate x_i into cell c*'s sum.
//
// Compute-bound pass: m*D flop per loaded element. To trade a statistically
// negligible amount of mean accuracy for large speedups, `subsample` > 1
// processes every subsample-th CONTIGUOUS TILE of rows (systematic tile
// sampling: coalesced, unbiased across the sequence; cell means over
// thousands of members are insensitive to 1/subsample thinning; rare cells
// degrade gracefully to their seed row via the finalize fallback).
//
// Seeds + accumulators both live in SMEM when they fit (2*m*D*4 bytes + m
// counts); wrapper falls back to global-atomic mode otherwise.
//
// Grid: (BH, ntiles_processed); Block: LM_BLK (8 warps), 1 row per warp per step.

template <typename scalar_t, int D, bool SMEM_ACC>
__global__ void lm_assign_kernel(
    const scalar_t* __restrict__ x,   // (BH, N, D)
    const float* __restrict__ seed_rows,   // (BH, m, D)
    const float* __restrict__ half_norms,  // (BH, m)
    float* __restrict__ acc,          // (BH, m, D) pre-zeroed
    int*   __restrict__ cnt,          // (BH, m)    pre-zeroed
    int*   __restrict__ assign_out,   // (BH, N) per-row cell id, or nullptr
    int N, int m, int tile_rows, int subsample
) {
    constexpr int K = D / 32;
    const int bh = blockIdx.x;
    const int tile = blockIdx.y * subsample;          // systematic tile sampling
    const int row0 = tile * tile_rows;
    if (row0 >= N) return;
    const int row1 = min(row0 + tile_rows, N);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int nwarp = LM_BLK / 32;
    const int mpl = (m + 31) / 32;                    // seeds per lane

    // Seeds are stored with stride D+1: in the hot dot-product loop the 32
    // lanes of a warp read 32 DIFFERENT seeds at the same column b, i.e.
    // addresses c*(stride)+b. With stride D (0 mod 32) that is a 32-way bank
    // conflict on every access; D+1 makes consecutive seeds land on
    // consecutive banks (conflict-free).
    constexpr int SP = D + 1;
    extern __shared__ float sm[];
    float* sS = sm;                                    // m*SP padded seeds
    float* sH = sm + m * SP;                           // m half-norms
    float* sA = SMEM_ACC ? (sm + m * SP + m) : nullptr; // m*D accumulators
    int*   sC = SMEM_ACC ? reinterpret_cast<int*>(sA + m * D) : nullptr;

    {
        const float* Sb = seed_rows + static_cast<size_t>(bh) * m * D;
        const float* Hb = half_norms + static_cast<size_t>(bh) * m;
        for (int e = threadIdx.x; e < m * D; e += LM_BLK)
            sS[(e / D) * SP + (e % D)] = Sb[e];
        if (SMEM_ACC)
            for (int e = threadIdx.x; e < m * D; e += LM_BLK) sA[e] = 0.0f;
        for (int e = threadIdx.x; e < m; e += LM_BLK) {
            sH[e] = Hb[e];
            if (SMEM_ACC) sC[e] = 0;
        }
    }
    __syncthreads();

    const scalar_t* xb = x + static_cast<size_t>(bh) * N * D;
    float* accb = acc + static_cast<size_t>(bh) * m * D;
    int*   cntb = cnt + static_cast<size_t>(bh) * m;

    for (int r = row0 + warp; r < row1; r += nwarp) {
        float xr[K];
        #pragma unroll
        for (int k = 0; k < K; k++)
            xr[k] = to_float(xb[static_cast<size_t>(r) * D + k * 32 + lane]);

        // per-lane best over its mpl seeds; dot via shuffle-broadcast of x.
        // F2: the shuffle is UNCONDITIONAL (all lanes participate); only the
        // per-lane FMA is guarded by `valid`, so m need not be a multiple of 32.
        float best = -CUDART_INF_F; int bc = -1;
        for (int cc = 0; cc < mpl; cc++) {
            const int c = cc * 32 + lane;
            const bool valid = (c < m);
            float dot = valid ? -sH[c] : -CUDART_INF_F;
            const float* sc = valid ? (sS + c * SP) : nullptr;
            #pragma unroll
            for (int kb = 0; kb < K; kb++)
                for (int l = 0; l < 32; l++) {
                    const float xv = __shfl_sync(0xffffffffu, xr[kb], l);
                    if (valid) dot = fmaf(sc[kb * 32 + l], xv, dot);
                }
            if (dot > best) { best = dot; bc = c; }
        }
        // warp argmax
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, best, o);
            const int   oc = __shfl_down_sync(0xffffffffu, bc,   o);
            if (ov > best) { best = ov; bc = oc; }
        }
        bc = __shfl_sync(0xffffffffu, bc, 0);

        // per-row cell id for the straight-through backward (membership fixed).
        // Only processed tiles are written; unprocessed rows (subsample>1) keep
        // their -1 init and are skipped by the backward scatter.
        if (assign_out && lane == 0) assign_out[static_cast<size_t>(bh) * N + r] = bc;

        // accumulate row into cell bc
        if (SMEM_ACC) {
            #pragma unroll
            for (int k = 0; k < K; k++) atomicAdd(&sA[bc * D + k * 32 + lane], xr[k]);
            if (lane == 0) atomicAdd(&sC[bc], 1);
        } else {
            #pragma unroll
            for (int k = 0; k < K; k++) atomicAdd(&accb[bc * D + k * 32 + lane], xr[k]);
            if (lane == 0) atomicAdd(&cntb[bc], 1);
        }
    }

    if (SMEM_ACC) {
        __syncthreads();
        for (int e = threadIdx.x; e < m * D; e += LM_BLK)
            if (sA[e] != 0.0f) atomicAdd(&accb[e], sA[e]);
        for (int e = threadIdx.x; e < m; e += LM_BLK)
            if (sC[e] != 0) atomicAdd(&cntb[e], sC[e]);
    }
}

// ========================================================== 6. finalize kernel
//
// x_tilde[c] = (cnt[c] > 0 ? acc[c]/cnt[c] : seed_row[c]) * scale
// Grid: (BH, m); Block: D.

template <typename scalar_t, int D>
__global__ void lm_finalize_kernel(
    const float* __restrict__ acc,        // (BH, m, D)
    const int*   __restrict__ cnt,        // (BH, m)
    const float* __restrict__ seed_rows,  // (BH, m, D)
    scalar_t* __restrict__ x_tilde,       // (BH, m, D)
    int m, float scale
) {
    const int bh = blockIdx.x, c = blockIdx.y, t = threadIdx.x;
    const size_t off = (static_cast<size_t>(bh) * m + c) * D + t;
    const int n = cnt[static_cast<size_t>(bh) * m + c];
    const float v = (n > 0) ? acc[off] / (float)n : seed_rows[off];
    x_tilde[off] = from_float<scalar_t>(v * scale);
}

// =============================================================== workspace

struct LmWorkspace {
    float* gram;        // BH * D * D
    float* minv;        // BH * D * D
    float* flr;         // BH
    float* gscore;      // BH * N
    float* cand_v;      // BH * topm_blocks * m
    int*   cand_i;      // BH * topm_blocks * m
    int*   seeds;       // BH * m
    float* seed_rows;   // BH * m * D
    float* half_norms;  // BH * m
    float* acc;         // BH * m * D
    int*   cnt;         // BH * m
};

// F3: topm_blocks derivation lives in one place so the workspace-size function
// and the launch wrapper cannot disagree.
inline int lm_topm_blocks(int N, int nsm) {
    return min(4 * nsm, max(1, N / (4 * LM_BLK)));
}

inline size_t lm_workspace_bytes(int BH, int N, int D, int m, int topm_blocks) {
    size_t s = 0;
    s += (size_t)BH * D * D * 4 * 2;          // gram + minv
    s += (size_t)BH * 4;                      // flr
    s += (size_t)BH * N * 4;                  // gscore
    s += (size_t)BH * topm_blocks * m * 8;    // cand_v + cand_i
    s += (size_t)BH * m * 4;                  // seeds
    s += (size_t)BH * m * D * 4 * 2;          // seed_rows + acc
    s += (size_t)BH * m * 4 * 2;              // half_norms + cnt
    return s + 1024;                          // alignment slack
}

// F3: caller-facing overload -- derives topm_blocks from the current device so
// the caller need not replicate the SM-count query and the min/max formula.
inline size_t lm_workspace_bytes(int BH, int N, int D, int m) {
    int dev = 0, nsm = 1;
    if (cudaGetDevice(&dev) == cudaSuccess)
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    return lm_workspace_bytes(BH, N, D, m, lm_topm_blocks(N, nsm));
}

// carve the workspace out of a caller-provided buffer (16B-aligned pieces)
inline LmWorkspace lm_carve_workspace(void* buf, int BH, int N, int D, int m, int topm_blocks) {
    auto align16 = [](char*& p, size_t bytes) {
        char* q = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(p) + 15) & ~uintptr_t(15));
        p = q + bytes;
        return q;
    };
    char* p = static_cast<char*>(buf);
    LmWorkspace w;
    w.gram       = reinterpret_cast<float*>(align16(p, (size_t)BH * D * D * 4));
    w.minv       = reinterpret_cast<float*>(align16(p, (size_t)BH * D * D * 4));
    w.flr        = reinterpret_cast<float*>(align16(p, (size_t)BH * 4));
    w.gscore     = reinterpret_cast<float*>(align16(p, (size_t)BH * N * 4));
    w.cand_v     = reinterpret_cast<float*>(align16(p, (size_t)BH * topm_blocks * m * 4));
    w.cand_i     = reinterpret_cast<int*>  (align16(p, (size_t)BH * topm_blocks * m * 4));
    w.seeds      = reinterpret_cast<int*>  (align16(p, (size_t)BH * m * 4));
    w.seed_rows  = reinterpret_cast<float*>(align16(p, (size_t)BH * m * D * 4));
    w.half_norms = reinterpret_cast<float*>(align16(p, (size_t)BH * m * 4));
    w.acc        = reinterpret_cast<float*>(align16(p, (size_t)BH * m * D * 4));
    w.cnt        = reinterpret_cast<int*>  (align16(p, (size_t)BH * m * 4));
    return w;
}

// ============================================================ launch wrapper
//
// Computes X_tilde (BH, m, D) = scale * leverage-seeded Voronoi-mean landmarks
// of X (BH, N, D). Call once for K (-> K_tilde) and once for Q (-> Q_tilde),
// with different seeds. No host synchronization; all sizing is host-side.
//
//   subsample: 1 = exact means; s>1 processes 1/s of row tiles in the assign
//              pass (systematic tile sampling). 4 is a good default at N >= 1M.
//   rng_seed:  fixed seed => fully deterministic landmark selection.

template <typename scalar_t, int D>
void launch_rls_vmean_landmarks(
    const scalar_t* x, scalar_t* x_tilde,
    int BH, int N, int m, float scale,
    void* workspace, size_t workspace_bytes,
    uint64_t rng_seed, int subsample,
    cudaStream_t stream,
    int* assign_out = nullptr,   // (BH, N) per-row cell id for the backward, or null
    int* cnt_out = nullptr       // (BH, m) per-cell count for the backward, or null
) {
    static_assert(D == 64 || D == 128, "supported head dims: 64, 128");
    FN_CHECK(BH > 0 && N > 0 && m > 0, "launch_rls_vmean_landmarks: invalid dims");
    FN_CHECK(m <= LM_TOPM_MAX, "m exceeds LM_TOPM_MAX");
    FN_CHECK(m + LM_BLK <= LM_SORT_P2, "LM_SORT_P2 too small for m + LM_BLK");
    FN_CHECK(m <= N, "m > N");
    FN_CHECK(subsample >= 1, "subsample must be >= 1");

    int dev = 0, nsm = 0, max_smem = 0;
    FN_CUDA_CHECK(cudaGetDevice(&dev));
    FN_CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    FN_CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));

    // enough stage-A blocks to fill the machine, few enough that stage B is tiny
    const int topm_blocks = lm_topm_blocks(N, nsm);

    FN_CHECK(workspace_bytes >= lm_workspace_bytes(BH, N, D, m, topm_blocks),
             "workspace too small");
    LmWorkspace w = lm_carve_workspace(workspace, BH, N, D, m, topm_blocks);

    FN_CUDA_CHECK(cudaMemsetAsync(w.gram, 0, (size_t)BH * D * D * 4, stream));
    FN_CUDA_CHECK(cudaMemsetAsync(w.acc,  0, (size_t)BH * m * D * 4, stream));
    FN_CUDA_CHECK(cudaMemsetAsync(w.cnt,  0, (size_t)BH * m * 4, stream));

    // ---- 1. gram
    {
        const int tiles_wanted = max(1, (4 * nsm) / BH);
        const int rows_per_block = max(LM_GRAM_TILE,
            ((N + tiles_wanted - 1) / tiles_wanted + LM_GRAM_TILE - 1) / LM_GRAM_TILE * LM_GRAM_TILE);
        dim3 grid(BH, (N + rows_per_block - 1) / rows_per_block);
        const size_t smem = 2u * LM_GRAM_TILE * D * sizeof(float);
        lm_gram_kernel<scalar_t, D><<<grid, LM_BLK, smem, stream>>>(x, w.gram, N, rows_per_block);
        FN_CUDA_KERNEL_CHECK();
    }
    // ---- 2. prep (M, floor)
    {
        const size_t smem = (2u * D * D + D) * sizeof(float);
        // F1: opt in to > 48 KB dynamic SMEM (D=128 needs ~132 KB).
        if (smem > 48u * 1024) {
            FN_CHECK((int)smem <= max_smem,
                     "lm_prep_kernel SMEM exceeds device limit (D=128 needs a tiled layout on this GPU)");
            FN_CUDA_CHECK(cudaFuncSetAttribute((const void*)lm_prep_kernel<D>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        }
        lm_prep_kernel<D><<<BH, D, smem, stream>>>(w.gram, w.minv, w.flr, N, m);
        FN_CUDA_KERNEL_CHECK();
    }
    // ---- 3. scores + gumbel
    {
        const int tiles_wanted = max(1, (4 * nsm) / BH);
        const int rows_per_block = max(1, (N + tiles_wanted - 1) / tiles_wanted);
        dim3 grid(BH, (N + rows_per_block - 1) / rows_per_block);
        const size_t smem = (size_t)D * D * sizeof(float);
        // F1: opt in to > 48 KB dynamic SMEM (D=128 needs 64 KB).
        if (smem > 48u * 1024) {
            FN_CHECK((int)smem <= max_smem, "lm_score_kernel SMEM exceeds device limit");
            FN_CUDA_CHECK(cudaFuncSetAttribute((const void*)lm_score_kernel<scalar_t, D>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        }
        lm_score_kernel<scalar_t, D><<<grid, LM_BLK, smem, stream>>>(
            x, w.minv, w.flr, w.gscore, N, rows_per_block,
            (uint32_t)(rng_seed & 0xffffffffu), (uint32_t)(rng_seed >> 32));
        FN_CUDA_KERNEL_CHECK();
    }
    // ---- 4. top-m (seeds)
    {
        dim3 gA(BH, topm_blocks);
        lm_topm_stageA_kernel<<<gA, LM_BLK, 0, stream>>>(w.gscore, w.cand_v, w.cand_i, N, m);
        FN_CUDA_KERNEL_CHECK();
        lm_topm_stageB_kernel<<<BH, LM_BLK, 0, stream>>>(w.cand_v, w.cand_i, w.seeds, topm_blocks, m);
        FN_CUDA_KERNEL_CHECK();
    }
    // ---- 5. gather seeds, assign + accumulate
    {
        dim3 gg(BH, m);
        lm_seed_gather_kernel<scalar_t, D><<<gg, D, 0, stream>>>(
            x, w.seeds, w.seed_rows, w.half_norms, N, m);
        FN_CUDA_KERNEL_CHECK();

        // -1 init so the backward can tell processed rows (subsample>1 leaves
        // some tiles unwritten). 0xFF bytes == int -1.
        if (assign_out)
            FN_CUDA_CHECK(cudaMemsetAsync(assign_out, 0xFF, (size_t)BH * N * 4, stream));

        const int tile_rows = 4 * LM_BLK;
        const int ntiles = (N + tile_rows - 1) / tile_rows;
        const int nproc = (ntiles + subsample - 1) / subsample;
        const size_t smem_base = ((size_t)m * (D + 1) + m) * sizeof(float);
        const size_t smem_acc  = smem_base + (size_t)m * D * sizeof(float) + m * sizeof(int);
        dim3 ga(BH, nproc);
        if (smem_acc <= (size_t)max_smem) {
            if (smem_acc > 48u * 1024)
                FN_CUDA_CHECK(cudaFuncSetAttribute(
                    (const void*)lm_assign_kernel<scalar_t, D, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_acc));
            lm_assign_kernel<scalar_t, D, true><<<ga, LM_BLK, smem_acc, stream>>>(
                x, w.seed_rows, w.half_norms, w.acc, w.cnt, assign_out, N, m, tile_rows, subsample);
        } else {
            if (smem_base > 48u * 1024)
                FN_CUDA_CHECK(cudaFuncSetAttribute(
                    (const void*)lm_assign_kernel<scalar_t, D, false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_base));
            lm_assign_kernel<scalar_t, D, false><<<ga, LM_BLK, smem_base, stream>>>(
                x, w.seed_rows, w.half_norms, w.acc, w.cnt, assign_out, N, m, tile_rows, subsample);
        }
        FN_CUDA_KERNEL_CHECK();
        // export the final per-cell counts for the backward (counts are >=1:
        // every seed assigns to its own cell, so no empty cells among seeds).
        if (cnt_out)
            FN_CUDA_CHECK(cudaMemcpyAsync(cnt_out, w.cnt, (size_t)BH * m * 4,
                                          cudaMemcpyDeviceToDevice, stream));
    }
    // ---- 6. finalize (means, empty-cell fallback, fold scale)
    {
        dim3 gf(BH, m);
        lm_finalize_kernel<scalar_t, D><<<gf, D, 0, stream>>>(
            w.acc, w.cnt, w.seed_rows, x_tilde, m, scale);
        FN_CUDA_KERNEL_CHECK();
    }
}

} // namespace flash_nystrom
