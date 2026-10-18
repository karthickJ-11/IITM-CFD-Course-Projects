// cuda_cpu_shim.h -- minimal CPU emulation of the CUDA runtime features used by
// Lid_driven_Cavity.cu, so that the UNMODIFIED kernels can be executed and verified on a machine
// without an NVIDIA GPU. Each kernel launch is executed serially over all blocks and threads.
// This is valid for this code because no kernel has intra-launch data dependencies except the
// red-black Gauss-Seidel kernel, whose one-colour-per-launch design makes serial and parallel
// execution identical. build: tools/emulate.sh (rewrites kernel<<<g,b>>>(...) into LAUNCH(...)).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
struct dim3 { unsigned x, y, z; dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
struct uint3_emu { unsigned x, y, z; };
static uint3_emu threadIdx, blockIdx; static dim3 blockDim, gridDim;
#define __global__
#define __device__
#define __host__
typedef int cudaError_t;
#define cudaSuccess 0
enum { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3, cudaMemcpyDefault = 4 };
static inline const char *cudaGetErrorString(cudaError_t) { return "no error"; }
template <class T> static inline cudaError_t cudaMallocManaged(T **p, size_t n) { *p = (T *)malloc(n); return *p ? 0 : 1; }
static inline cudaError_t cudaFree(void *p) { free(p); return 0; }
static inline cudaError_t cudaDeviceSynchronize() { return 0; }
static inline cudaError_t cudaMemcpy(void *d, const void *s, size_t n, int) { memcpy(d, s, n); return 0; }
#define LAUNCH(kernel, G, B, ...) do { dim3 g_ = (G), b_ = (B); gridDim = g_; blockDim = b_;            \
    for (blockIdx.y = 0; blockIdx.y < g_.y; blockIdx.y++) for (blockIdx.x = 0; blockIdx.x < g_.x; blockIdx.x++) \
    for (threadIdx.y = 0; threadIdx.y < b_.y; threadIdx.y++) for (threadIdx.x = 0; threadIdx.x < b_.x; threadIdx.x++) \
        kernel(__VA_ARGS__); } while (0)
