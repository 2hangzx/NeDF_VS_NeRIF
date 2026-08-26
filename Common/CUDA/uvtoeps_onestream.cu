/*-------------------------------------------------------------------------
 *
 * CUDA functions for uv to eps
 *
 *
 * CODE by       Wei Hu
---------------------------------------------------------------------------
---------------------------------------------------------------------------
Copyright (c) 2024, Beihang University
All rights reserved.
Contact: Weihu22@buaa.edu.cn
Codes  : XXXX
---------------------------------------------------------------------------
 */
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <chrono>
#include <iostream>

#include "BOSLAB_common.h"
#include "uvtoeps.h"
#include "gpu_buffer_pool.h"

#define cudaCheckErrors(msg) \
do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
				mexPrintf("Error: %s\n", msg);\
				mexPrintf("uvtoeps: %s\n", cudaGetErrorString(__err)); \
        } \
} while (0)

 // ------------------------- device utils ------------------------------
__device__ void matmul3x3(const float* a, const float* b, float* c) {
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			c[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
}

__device__ void transpose3x3(const float* a, float* at) {
	at[0] = a[0]; at[1] = a[3]; at[2] = a[6];
	at[3] = a[1]; at[4] = a[4]; at[5] = a[7];
	at[6] = a[2]; at[7] = a[5]; at[8] = a[8];
}

__device__ void inv3x3(const float* a, float* inva) {
	float det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
	if (det == 0) return;
	float invdet = 1.0f / det;
	inva[0] = (a[4] * a[8] - a[5] * a[7])*invdet;
	inva[1] = (a[2] * a[7] - a[1] * a[8])*invdet;
	inva[2] = (a[1] * a[5] - a[2] * a[4])*invdet;
	inva[3] = (a[5] * a[6] - a[3] * a[8])*invdet;
	inva[4] = (a[0] * a[8] - a[2] * a[6])*invdet;
	inva[5] = (a[2] * a[3] - a[0] * a[5])*invdet;
	inva[6] = (a[3] * a[7] - a[4] * a[6])*invdet;
	inva[7] = (a[1] * a[6] - a[0] * a[7])*invdet;
	inva[8] = (a[0] * a[4] - a[1] * a[3])*invdet;
}

// ------------------------- global utils ------------------------------

__global__ void precompute_KRinv_kernel(
	const float* imcam,
	const float* rrcam,
	const float* zbc,
	const float* zpc,
	float* kr_inv_out,
	float* scale_out,
	int numcam)
{
	int camidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (camidx >= numcam) return;

	float k[9], r[9], kt[9], rt[9], kr[9], kr_inv[9];
#pragma unroll
	for (int i = 0; i < 9; i++) {
		k[i] = imcam[camidx * 9 + i];
		r[i] = rrcam[camidx * 9 + i];
	}

	transpose3x3(k, kt);
	transpose3x3(r, rt);
	matmul3x3(kt, rt, kr);
	inv3x3(kr, kr_inv);

#pragma unroll
	for (int i = 0; i < 9; i++)
		kr_inv_out[camidx * 9 + i] = kr_inv[i];

	float zb = zbc[camidx];
	float zp = zpc[camidx];
	scale_out[camidx] = zb / (zb - zp);
}

__global__ void uvtoeps_kernel_opt(
	const float* projectionsu,
	const float* projectionsv,
	float* projectionsob,
	const float* kr_inv,
	const float* scale,
	int numcam, int rows, int cols)
{
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int camidx = blockIdx.z;
	if (row >= rows || col >= cols || camidx >= numcam) return;

	// shared memory for per-block camera
	__shared__ float s_krinv[9];
	__shared__ float s_scale;
	if (threadIdx.x == 0 && threadIdx.y == 0) {
#pragma unroll
		for (int i = 0; i < 9; i++) s_krinv[i] = kr_inv[camidx * 9 + i];
		s_scale = scale[camidx];
	}
	__syncthreads();

	int pixelindex = camidx * rows*cols + row * cols + col;
	float u = -projectionsu[pixelindex];
	float v = -projectionsv[pixelindex];

	float eps0 = (s_krinv[0] * u + s_krinv[1] * v) * s_scale;
	float eps1 = (s_krinv[3] * u + s_krinv[4] * v) * s_scale;
	float eps2 = (s_krinv[6] * u + s_krinv[7] * v) * s_scale;

	int stride = rows * cols*numcam;
	projectionsob[pixelindex] = eps0;
	projectionsob[pixelindex + stride] = eps1;
	projectionsob[pixelindex + 2 * stride] = eps2;
}

//______________________________________________________________________________
//
//      Function:       voxel_backprojection
//
//      Description:    Main host function for FDK backprojection (invokes the kernel)
//______________________________________________________________________________

void uvtoeps(Geometry geo, float* projectionsU, float* projectionsV, float* result, const GpuIds& gpuids, GPUBufferPool& gpuPool) {
	auto t0 = std::chrono::high_resolution_clock::now();
	// Prepare for MultiGPU

	cudaSetDevice(0);



	// 定义网格和块的大小

	int threads_cam = 128;
	int blocks_cam =
		(geo.numCam + threads_cam - 1) / threads_cam;
	dim3 block_uv(16, 16);
	dim3 grid_uv(
		(geo.maxnCamU + block_uv.x - 1) / block_uv.x,
		(geo.maxnCamV + block_uv.y - 1) / block_uv.y,
		geo.numCam);

	int buf = 0;

	cudaMemcpy(gpuPool.d_U[buf], projectionsU, gpuPool.projSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuPool.d_V[buf], projectionsV, gpuPool.projSize * sizeof(float), cudaMemcpyHostToDevice);


	precompute_KRinv_kernel << <blocks_cam, threads_cam >> > (
		gpuPool.d_IMCam,
		gpuPool.d_RrCam,
		gpuPool.d_Zbc,
		gpuPool.d_Zpc,
		gpuPool.d_KR_inv,
		gpuPool.d_scale,
		geo.numCam);
	cudaCheckErrors("precompute_KRinv_kernel failed");

	// 调用核函数
	uvtoeps_kernel_opt << <grid_uv, block_uv >> > (
		gpuPool.d_U[buf],
		gpuPool.d_V[buf],
		gpuPool.d_projectionsob,
		gpuPool.d_KR_inv,
		gpuPool.d_scale,
		geo.numCam,
		geo.maxnCamU,
		geo.maxnCamV);

	cudaCheckErrors("uvtoeps_kernel_opt failed");

	cudaDeviceSynchronize();

	auto t1 = std::chrono::high_resolution_clock::now();
	double uvtoeps_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

	mexPrintf("uvtoeps time = %.3f ms\n", uvtoeps_ms);

	// 将结果从设备复制回主机
	cudaMemcpy(result, gpuPool.d_projectionsob, 3 * geo.numCam * geo.maxnCamU * geo.maxnCamV * sizeof(float), cudaMemcpyDeviceToHost);


}

///*-------------------------------------------------------------------------
// *
// * CUDA functions for uv to eps
// *
// *
// * CODE by       Wei Hu (optimized)
//---------------------------------------------------------------------------*/
//
//#include <algorithm>
//#include <cuda_runtime_api.h>
//#include <cuda.h>
//#include <math.h>
//#include <device_launch_parameters.h>
//
//#include "E:/github_upload/boslab-v2/Common/CUDA/BOSLAB_common.h"
//#include "E:/github_upload/boslab-v2/Common/CUDA/uvtoeps.h"
//
//#define cudaCheckErrors(msg) \
//do { \
//    cudaError_t __err = cudaGetLastError(); \
//    if (__err != cudaSuccess) { \
//        mexPrintf("Error: %s\n", msg);\
//        mexPrintf("uvtoeps: %s\n", cudaGetErrorString(__err)); \
//    } \
//} while (0)
//
//// ------------------------- device utils ------------------------------
//__device__ void matMul3x3(const float* A, const float* B, float* C) {
//	for (int i = 0; i < 3; i++)
//		for (int j = 0; j < 3; j++)
//			C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
//}
//
//__device__ void transpose3x3(const float* A, float* At) {
//	At[0] = A[0]; At[1] = A[3]; At[2] = A[6];
//	At[3] = A[1]; At[4] = A[4]; At[5] = A[7];
//	At[6] = A[2]; At[7] = A[5]; At[8] = A[8];
//}
//
//__device__ void inv3x3(const float* A, float* invA) {
//	float det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
//	if (det == 0) return;
//	float invDet = 1.0f / det;
//	invA[0] = (A[4] * A[8] - A[5] * A[7])*invDet;
//	invA[1] = (A[2] * A[7] - A[1] * A[8])*invDet;
//	invA[2] = (A[1] * A[5] - A[2] * A[4])*invDet;
//	invA[3] = (A[5] * A[6] - A[3] * A[8])*invDet;
//	invA[4] = (A[0] * A[8] - A[2] * A[6])*invDet;
//	invA[5] = (A[2] * A[3] - A[0] * A[5])*invDet;
//	invA[6] = (A[3] * A[7] - A[4] * A[6])*invDet;
//	invA[7] = (A[1] * A[6] - A[0] * A[7])*invDet;
//	invA[8] = (A[0] * A[4] - A[1] * A[3])*invDet;
//}
//
//// ------------------------- camera-level kernel -----------------------
//__global__ void precompute_KRinv_kernel(
//	const float* IMCam,
//	const float* RrCam,
//	const float* Zbc,
//	const float* Zpc,
//	float* KR_inv_out,
//	float* scale_out,
//	int numCam)
//{
//	int camIdx = blockIdx.x * blockDim.x + threadIdx.x;
//	if (camIdx >= numCam) return;
//
//	float K[9], R[9], KT[9], RT[9], KR[9], KR_inv[9];
//#pragma unroll
//	for (int i = 0; i < 9; i++) {
//		K[i] = IMCam[camIdx * 9 + i];
//		R[i] = RrCam[camIdx * 9 + i];
//	}
//
//	transpose3x3(K, KT);
//	transpose3x3(R, RT);
//	matMul3x3(KT, RT, KR);
//	inv3x3(KR, KR_inv);
//
//#pragma unroll
//	for (int i = 0; i < 9; i++)
//		KR_inv_out[camIdx * 9 + i] = KR_inv[i];
//
//	float Zb = Zbc[camIdx];
//	float Zp = Zpc[camIdx];
//	scale_out[camIdx] = Zb / (Zb - Zp);
//}
//
//// ------------------------- pixel-level kernel ------------------------
//__global__ void uvtoeps_kernel_opt(
//	const float* projectionsU,
//	const float* projectionsV,
//	float* projectionsob,
//	const float* KR_inv,
//	const float* scale,
//	int numCam, int rows, int cols)
//{
//	int col = blockIdx.x*blockDim.x + threadIdx.x;
//	int row = blockIdx.y*blockDim.y + threadIdx.y;
//	int camIdx = blockIdx.z;
//	if (row >= rows || col >= cols || camIdx >= numCam) return;
//
//	// shared memory for per-block camera
//	__shared__ float s_KRinv[9];
//	__shared__ float s_scale;
//	if (threadIdx.x == 0 && threadIdx.y == 0) {
//#pragma unroll
//		for (int i = 0; i < 9; i++) s_KRinv[i] = KR_inv[camIdx * 9 + i];
//		s_scale = scale[camIdx];
//	}
//	__syncthreads();
//
//	int pixelIndex = camIdx * rows*cols + row * cols + col;
//	float u = -projectionsU[pixelIndex];
//	float v = -projectionsV[pixelIndex];
//
//	float eps0 = (s_KRinv[0] * u + s_KRinv[1] * v) * s_scale;
//	float eps1 = (s_KRinv[3] * u + s_KRinv[4] * v) * s_scale;
//	float eps2 = (s_KRinv[6] * u + s_KRinv[7] * v) * s_scale;
//
//	int stride = rows * cols*numCam;
//	projectionsob[pixelIndex] = eps0;
//	projectionsob[pixelIndex + stride] = eps1;
//	projectionsob[pixelIndex + 2 * stride] = eps2;
//}
//
//// ------------------------- host function ----------------------------
//void uvtoeps(Geometry geo, float* projectionsU, float* projectionsV, float* result, const GpuIds& gpuids) {
//
//	int deviceCount = gpuids.GetLength();
//	cudaCheckErrors("Device query fail");
//	if (deviceCount == 0) {
//		mexPrintf("uvtoeps:GPUsemexPrintflect\n", "There are no available device(s) that support CUDA\n");
//	}
//	if (!gpuids.AreEqualDevices()) {
//		mexPrintf("uvtoeps:GPUselect\n", "Detected one (or more) different GPUs.\n");
//	}
//	cudaSetDevice(0);
//
//	float *d_projectionsU, *d_projectionsV, *d_projectionsob;
//	float *d_IMCam, *d_RrCam, *d_Zbc, *d_Zpc;
//	float *d_KR_inv, *d_scale;
//
//	// ---------------- allocate device memory -----------------
//	cudaMalloc((void**)&d_projectionsU, geo.maxnCamU*geo.maxnCamV*geo.numCam * sizeof(float));
//	cudaMalloc((void**)&d_projectionsV, geo.maxnCamU*geo.maxnCamV*geo.numCam * sizeof(float));
//	cudaMalloc((void**)&d_IMCam, 9 * geo.numCam * sizeof(float));
//	cudaMalloc((void**)&d_RrCam, 9 * geo.numCam * sizeof(float));
//	cudaMalloc((void**)&d_Zbc, geo.numCam * sizeof(float));
//	cudaMalloc((void**)&d_Zpc, geo.numCam * sizeof(float));
//	cudaMalloc((void**)&d_projectionsob, 3 * geo.numCam*geo.maxnCamU*geo.maxnCamV * sizeof(float));
//	cudaMalloc((void**)&d_KR_inv, 9 * geo.numCam * sizeof(float));
//	cudaMalloc((void**)&d_scale, geo.numCam * sizeof(float));
//
//	// ---------------- copy to device -----------------
//	cudaMemcpy(d_projectionsU, projectionsU, geo.maxnCamU*geo.maxnCamV*geo.numCam * sizeof(float), cudaMemcpyHostToDevice);
//	cudaMemcpy(d_projectionsV, projectionsV, geo.maxnCamU*geo.maxnCamV*geo.numCam * sizeof(float), cudaMemcpyHostToDevice);
//	cudaMemcpy(d_IMCam, geo.IMCam, 9 * geo.numCam * sizeof(float), cudaMemcpyHostToDevice);
//	cudaMemcpy(d_RrCam, geo.RrCam, 9 * geo.numCam * sizeof(float), cudaMemcpyHostToDevice);
//	cudaMemcpy(d_Zbc, geo.Zbc, geo.numCam * sizeof(float), cudaMemcpyHostToDevice);
//	cudaMemcpy(d_Zpc, geo.Zpc, geo.numCam * sizeof(float), cudaMemcpyHostToDevice);
//
//	// ---------------- precompute KR_inv -----------------
//	int threads = 128;
//	int blocks = (geo.numCam + threads - 1) / threads;
//	precompute_KRinv_kernel << <blocks, threads >> > (
//		d_IMCam, d_RrCam, d_Zbc, d_Zpc, d_KR_inv, d_scale, geo.numCam);
//	cudaCheckErrors("precompute_KRinv_kernel failed");
//
//	// ---------------- pixel kernel -----------------
//	dim3 blockSize(16, 16);
//	dim3 gridSize((geo.maxnCamU + blockSize.x - 1) / blockSize.x,
//		(geo.maxnCamV + blockSize.y - 1) / blockSize.y,
//		geo.numCam);
//
//	uvtoeps_kernel_opt << <gridSize, blockSize >> > (
//		d_projectionsU, d_projectionsV, d_projectionsob,
//		d_KR_inv, d_scale, geo.numCam, geo.maxnCamU, geo.maxnCamV);
//	cudaCheckErrors("uvtoeps_kernel_opt failed");
//
//	// ---------------- copy back -----------------
//	cudaMemcpy(result, d_projectionsob, 3 * geo.numCam*geo.maxnCamU*geo.maxnCamV * sizeof(float), cudaMemcpyDeviceToHost);
//
//	// ---------------- free memory -----------------
//	cudaFree(d_projectionsU);
//	cudaFree(d_projectionsV);
//	cudaFree(d_IMCam);
//	cudaFree(d_RrCam);
//	cudaFree(d_Zbc);
//	cudaFree(d_Zpc);
//	cudaFree(d_projectionsob);
//	cudaFree(d_KR_inv);
//	cudaFree(d_scale);
//}
