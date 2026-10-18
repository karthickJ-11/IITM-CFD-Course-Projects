// CPU emulation of the CUDA runtime subset used by the course-project codes.
// Every kernel launch  k<<<G, B>>>(args)  is rewritten by emulate.sh into  LAUNCH(k, G, B, args), which
// executes the UNMODIFIED kernel body once per (blockIdx, threadIdx), serially. Kernels whose threads
// write disjoint data (all kernels here; the pressure solver uses red-black ordering) therefore give the
// same result as on the GPU, apart from the summation order of atomicAdd and FMA contraction.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
struct uint3_emu { unsigned x,y,z; };
static uint3_emu threadIdx, blockIdx; static dim3 blockDim, gridDim;
#define __global__
#define __device__
#define __host__
typedef int cudaError_t;
#define cudaSuccess 0
enum { cudaMemcpyHostToHost=0, cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2, cudaMemcpyDeviceToDevice=3, cudaMemcpyDefault=4 };
static inline const char *cudaGetErrorString(cudaError_t){ return "no error"; }
static inline cudaError_t cudaGetLastError(){ return 0; }
template <class T> static inline cudaError_t cudaMallocManaged(T **p, size_t bytes){ *p=(T*)malloc(bytes); return *p?0:1; }
static inline cudaError_t cudaMalloc(void **p, size_t bytes){ *p=malloc(bytes); return *p?0:1; }
static inline cudaError_t cudaMemset(void *p, int value, size_t bytes){ memset(p, value, bytes); return 0; }
static inline cudaError_t cudaFree(void *p){ free(p); return 0; }
static inline cudaError_t cudaDeviceSynchronize(){ return 0; }
static inline cudaError_t cudaMemcpy(void *d,const void *s,size_t bytes,int){ memcpy(d,s,bytes); return 0; }
// events -> monotonic wall clock
struct cudaEventEmu { double ms; };
typedef cudaEventEmu *cudaEvent_t;
static inline cudaError_t cudaEventCreate(cudaEvent_t *e){ *e = new cudaEventEmu(); return 0; }
static inline cudaError_t cudaEventRecord(cudaEvent_t e, int = 0){ timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); e->ms = ts.tv_sec*1e3 + ts.tv_nsec*1e-6; return 0; }
static inline cudaError_t cudaEventSynchronize(cudaEvent_t){ return 0; }
static inline cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t e0, cudaEvent_t e1){ *ms = (float)(e1->ms - e0->ms); return 0; }
static inline cudaError_t cudaEventDestroy(cudaEvent_t e){ delete e; return 0; }
// atomics (serial execution -> plain read-modify-write)
static inline float atomicAdd(float *addr, float val){ float old=*addr; *addr=old+val; return old; }
static inline double atomicAdd(double *addr, double val){ double old=*addr; *addr=old+val; return old; }
static inline int atomicMax(int *addr, int val){ int old=*addr; if(val>old) *addr=val; return old; }
static inline int __float_as_int(float f){ int i; memcpy(&i,&f,sizeof i); return i; }
static inline float __int_as_float(int i){ float f; memcpy(&f,&i,sizeof f); return f; }
#define LAUNCH(kernel, G, B, ...) do { dim3 g_=(G), b_=(B); gridDim=g_; blockDim=b_; \
  for(blockIdx.y=0;blockIdx.y<g_.y;blockIdx.y++) for(blockIdx.x=0;blockIdx.x<g_.x;blockIdx.x++) \
  for(threadIdx.y=0;threadIdx.y<b_.y;threadIdx.y++) for(threadIdx.x=0;threadIdx.x<b_.x;threadIdx.x++) \
    kernel(__VA_ARGS__); } while(0)
