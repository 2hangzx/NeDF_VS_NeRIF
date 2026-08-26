/*-------------------------------------------------------------------------
 *
 * CUDA functions for img to uv
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
#include "imgtodisplacement.h"
#include "gpu_buffer_pool.h"


#define cudaCheckErrors(msg) \
do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
				mexPrintf("Error: %s\n", msg);\
				mexPrintf("imgtodisplacement: %s\n", cudaGetErrorString(__err)); \
        } \
} while (0)

__global__ void writeFlowKernel(
	const float* u,   // size H*W, row-major
	const float* v,
	float* Uout,      // MATLAB layout
	float* Vout,
	int V, int U,
	int k
) {
	int u_idx = blockIdx.x * blockDim.x + threadIdx.x; // x
	int v_idx = blockIdx.y * blockDim.y + threadIdx.y; // y

	if (u_idx >= U || v_idx >= V) return;

	int idx_cv = v_idx * U + u_idx;  // OpenCV row-major
	int idx_ml = v_idx
		+ u_idx * V
		+ k * (U * V);        // MATLAB column-major

	Uout[idx_ml] = u[idx_cv];
	Vout[idx_ml] = v[idx_cv];
}


void imgtodisplacement(const Geometry& geo,
	const char* filepath,
	const GpuIds& gpuids,
	float* projectionsU,
	float* projectionsV,
	GPUBufferPool& gpuPool) {
	auto t0 = std::chrono::high_resolution_clock::now();

	cudaSetDevice(0);
	
	// --------------- Farneback ----------------
	auto farn = cv::cuda::FarnebackOpticalFlow::create(
		3, 0.5, false, 16, 5, 5, 1.1, 0);

	// ---------------- PRE-ALLOCATE GpuMat ----------------
	dim3 block(16, 16);
	dim3 grid((geo.maxnCamU + 15) / 16, (geo.maxnCamV + 15) / 16);


	// =====================================================
	// ============== img -> displacement ==================
	// =====================================================
	
	int buf = 0;
	char filename[256];

	for (int k = 0; k < geo.numCam; ++k) {
		sprintf(filename, "%sparallelcomputing/iref/%04d.bmp", filepath, k + 1);
		//std::cout << "Reading iref: " << filename << std::endl;
		gpuPool.iref[k] = cv::imread(filename, cv::IMREAD_GRAYSCALE);
		sprintf(filename, "%sparallelcomputing/idis/%04d.bmp", filepath, k + 1);
		//std::cout << "Reading idis: " << filename << std::endl;
		gpuPool.idis[k] = cv::imread(filename, cv::IMREAD_GRAYSCALE);

		if (gpuPool.iref[k].empty() || gpuPool.idis[k].empty()) {
			mexPrintf("Warning: cannot load image cam %d\n", k);
			exit; // 或者 break / exit，根据需求
		}

	}

	for (int k = 0; k < geo.numCam; ++k)
	{
		gpuPool.d0[k].upload(gpuPool.iref[k], gpuPool.cvStreams[k]);
		gpuPool.d1[k].upload(gpuPool.idis[k], gpuPool.cvStreams[k]);

		farn->calc(gpuPool.d0[k], gpuPool.d1[k], gpuPool.d_flow[k], gpuPool.cvStreams[k]);

		cv::cuda::GpuMat planes[2] = { gpuPool.d_u[k], gpuPool.d_v[k] };
		cv::cuda::split(gpuPool.d_flow[k], planes, gpuPool.cvStreams[k]);

		writeFlowKernel << <grid, block, 0, gpuPool.streams[k] >> > (
			gpuPool.d_u[k].ptr<float>(),
			gpuPool.d_v[k].ptr<float>(),
			gpuPool.d_U[buf],
			gpuPool.d_V[buf],
			geo.maxnCamV,
			geo.maxnCamU,
			k);

		cudaEventRecord(gpuPool.events[k], gpuPool.streams[k]);
	}
	//cv::waitKey(1);
	// 等所有 camera
	for (int k = 0; k < geo.numCam; ++k)
		cudaStreamWaitEvent(gpuPool.flow_sync_stream,
			gpuPool.events[k], 0);

	// 同步流，保证 GPU 数据完整
	cudaStreamSynchronize(gpuPool.flow_sync_stream);


	cudaMemcpy(projectionsU, gpuPool.d_U[buf],
		gpuPool.projSize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(projectionsV, gpuPool.d_V[buf],
		gpuPool.projSize * sizeof(float), cudaMemcpyDeviceToHost);

	auto t1 = std::chrono::high_resolution_clock::now();
	double of_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

	mexPrintf("total OF time = %.3f ms\n", of_ms);
	
}
