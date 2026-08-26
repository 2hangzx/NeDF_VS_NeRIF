#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <atomic>

#include "BOSLAB_common.h"
#include "imgtodisplacement.h"
#include "reconstruction.h"

#pragma once

static constexpr int NUM_BUF = 2;

struct GPUBufferPool {
	const Geometry& geo;
	// host
	float* h_U_producer = nullptr;
	float* h_U = nullptr;
	float* h_V = nullptr;
	float* h_x = nullptr;
	float* h_p = nullptr;
	float* h_W = nullptr;
	float* h_image = nullptr;
	float* h_s_aux_1 = nullptr;
	float* h_s_aux_2 = nullptr;
	float* h_r_aux_2 = nullptr;
	float* h_s = nullptr;
	float* h_projectionsob = nullptr;
	//
	float* d_U[NUM_BUF];
	float* d_V[NUM_BUF];

	cudaEvent_t flow_frame_event[NUM_BUF];
	// iref idis
	std::vector<cv::Mat> iref;
	std::vector<cv::Mat> idis;
	// 当前 Producer / Consumer 使用的 buffer
	std::atomic<int> produce_idx{ 0 };
	std::atomic<int> consume_idx{ 0 };

	std::atomic<int> flow_version{ 0 };

	std::atomic<int> latest_ready_buf{ -1 };

	cudaStream_t flow_sync_stream;

	// img -> displacement buffers
	std::vector<cv::cuda::GpuMat> d0, d1, d_flow, d_u, d_v;

	// reconstruction buffers
	/*float* d_U = nullptr;
	float* d_V = nullptr;*/
	float* d_projectionsob = nullptr;
	float* d_IMCam = nullptr;
	float* d_RrCam = nullptr;
	float* d_Zbc = nullptr;
	float* d_Zpc = nullptr;
	float* d_KR_inv = nullptr;
	float* d_scale = nullptr;

	// IRN-TV-CGLS buffers
	float *d_x = nullptr, *d_W = nullptr;
	cudaArray* d_x_array = nullptr;
	cudaArray* d_p_array = nullptr;
	cudaTextureObject_t tex_x = 0;
	cudaTextureObject_t tex_p = 0;

	Point3D* projParamsArrayHost = nullptr;
	Point3D* projParamsArrayDev = nullptr;
	Point3D* projParamsArray2Dev = nullptr;
	Point3D* projParamsArray2Host = nullptr;

	Geometry* geoArray = nullptr;

	float* d_prox_aux_1 = nullptr;
	float* d_prox_aux_2 = nullptr;
	float* d_q_aux_1 = nullptr;
	float* d_q_aux_2 = nullptr;
	float* d_r_aux_1 = nullptr;
	float* d_r_aux_2 = nullptr;
	float* d_image = nullptr;
	float* d_p_aux_1 = nullptr;
	float* d_p_aux_2 = nullptr;
	float* d_s_aux_1 = nullptr;
	float* d_s_aux_2 = nullptr;
	float* d_p = nullptr;
	float* d_s = nullptr;
	float* d_gamma = nullptr;
	float* d_gamma1 = nullptr;
	float* d_aux_gamma = nullptr;
	float* d_gamma_q_aux_1 = nullptr;
	float* d_gamma_q_aux_2 = nullptr;
	float* d_alpha = nullptr;
	float* d_beta = nullptr;
	float* d_aux_m = nullptr;
	float* d_aux = nullptr;
	float* d_resL2 = nullptr;
	bool* d_condition = nullptr;
	bool* h_condition = nullptr;

	float* projCoeffArray2Dev = nullptr;
	float* projCoeffArray2Host = nullptr;
	
	size_t voxelSize;
	size_t projSize;

	// CUDA streams
	std::vector<cv::cuda::Stream> cvStreams;
	std::vector<cudaStream_t> streams;       // per-camera streams
	std::vector<cudaEvent_t> events;
	cudaStream_t reconstructionStream;
	cudaStream_t stream_diff[3];


	GPUBufferPool(const Geometry& geo_) : geo(geo_) {}

	void allocate(const Geometry& geo, int niter) {
		voxelSize = geo.nVoxelX * geo.nVoxelY * geo.nVoxelZ;
		projSize = geo.numCam * geo.maxnCamU * geo.maxnCamV;
		// host
		cudaMallocHost(&h_U_producer, projSize * sizeof(float));
		cudaMallocHost(&h_U, projSize * sizeof(float));
		cudaMallocHost(&h_V, projSize * sizeof(float));
		cudaMallocHost(&h_x, voxelSize * sizeof(float));
		cudaMallocHost(&h_p, voxelSize * sizeof(float));
		cudaMallocHost(&h_W, voxelSize * sizeof(float));
		cudaMallocHost(&h_image, 3 * voxelSize * sizeof(float));
		cudaMallocHost(&h_s_aux_1, voxelSize * sizeof(float));
		cudaMallocHost(&h_s_aux_2, voxelSize * sizeof(float));
		cudaMallocHost(&h_r_aux_2, 3*voxelSize * sizeof(float));
		cudaMallocHost(&h_s, voxelSize * sizeof(float));
		cudaMallocHost(&h_projectionsob, 3*projSize * sizeof(float));
		// iref idis
		iref.resize(geo.numCam);
		idis.resize(geo.numCam);
		// --------------- img -> displacement ---------------
		//cudaMalloc(&d_U, projSize * sizeof(float));
		//cudaMalloc(&d_V, projSize * sizeof(float));
		for (int i = 0; i < NUM_BUF; ++i) {
			cudaMalloc(&d_U[i], projSize * sizeof(float));
			cudaMalloc(&d_V[i], projSize * sizeof(float));

			cudaEventCreateWithFlags(&flow_frame_event[i],
				cudaEventDisableTiming);
		}

		cudaStreamCreate(&flow_sync_stream);



		d0.resize(geo.numCam);
		d1.resize(geo.numCam);
		d_flow.resize(geo.numCam);
		d_u.resize(geo.numCam);
		d_v.resize(geo.numCam);

		for (int k = 0; k < geo.numCam; ++k) {
			d0[k].create(geo.maxnCamV, geo.maxnCamU, CV_8UC1);
			d1[k].create(geo.maxnCamV, geo.maxnCamU, CV_8UC1);
			d_flow[k].create(geo.maxnCamV, geo.maxnCamU, CV_32FC2);
			d_u[k].create(geo.maxnCamV, geo.maxnCamU, CV_32F);
			d_v[k].create(geo.maxnCamV, geo.maxnCamU, CV_32F);
		}

		

		// ---------------- reconstruction buffers ----------------
		cudaMalloc(&d_projectionsob, 3 * projSize * sizeof(float));
		cudaMalloc(&d_IMCam, 9 * geo.numCam * sizeof(float));
		cudaMalloc(&d_RrCam, 9 * geo.numCam * sizeof(float));
		cudaMalloc(&d_Zbc, geo.numCam * sizeof(float));
		cudaMalloc(&d_Zpc, geo.numCam * sizeof(float));
		cudaMalloc(&d_KR_inv, 9 * geo.numCam * sizeof(float));
		cudaMalloc(&d_scale, geo.numCam * sizeof(float));

		cudaMemcpy(d_IMCam, geo.IMCam, 9 * geo.numCam * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_RrCam, geo.RrCam, 9 * geo.numCam * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Zbc, geo.Zbc, geo.numCam * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Zpc, geo.Zpc, geo.numCam * sizeof(float), cudaMemcpyHostToDevice);

		// ---------------- IRN-TV-CGLS ----------------
		geoArray = (Geometry*)malloc(sizeof(Geometry));
		geoArray[0] = geo;

		cudaMalloc(&d_x, voxelSize * sizeof(float));
		cudaMemset(d_x, 0, voxelSize * sizeof(float));

		cudaMalloc(&d_W, voxelSize * sizeof(float));
		cudaMalloc(&d_prox_aux_1, 3 * projSize * sizeof(float));//
		cudaMalloc(&d_prox_aux_2, 3 * voxelSize * sizeof(float));//
		cudaMalloc(&d_q_aux_1, 3 * projSize * sizeof(float));//
		cudaMalloc(&d_q_aux_2, 3 * voxelSize * sizeof(float));//
		cudaMalloc(&d_r_aux_1, 3 * projSize * sizeof(float));//
		cudaMalloc(&d_r_aux_2, 3 * voxelSize * sizeof(float));//
		cudaMalloc(&d_image, 3 * voxelSize * sizeof(float));//
		cudaMalloc(&d_p_aux_1, voxelSize * sizeof(float));//
		cudaMalloc(&d_p_aux_2, voxelSize * sizeof(float));//
		cudaMalloc(&d_s_aux_1, voxelSize * sizeof(float));//
		cudaMalloc(&d_s_aux_2, voxelSize * sizeof(float));//
		cudaMalloc(&d_p, voxelSize * sizeof(float));//
		cudaMalloc(&d_s, voxelSize * sizeof(float));//
		cudaMalloc(&d_gamma, sizeof(float));//
		cudaMalloc(&d_gamma1, sizeof(float));//
		cudaMalloc(&d_aux_gamma, sizeof(float));//
		cudaMalloc(&d_gamma_q_aux_1, sizeof(float));//
		cudaMalloc(&d_gamma_q_aux_2, sizeof(float));//
		cudaMalloc(&d_alpha, sizeof(float));//
		cudaMalloc(&d_beta, sizeof(float));//
		cudaMalloc(&d_aux_m, 3 * projSize * sizeof(float));//
		cudaMalloc(&d_aux, 3 * projSize * sizeof(float));//
		cudaMalloc(&d_resL2, niter * sizeof(float));//
		cudaMemset(d_resL2, 0, niter * sizeof(float));//
		cudaMalloc(&d_condition, sizeof(bool));//
		h_condition = (bool*)malloc(sizeof(bool));//
		//cudaMemset(d_x, 0, voxelSize * sizeof(float));
		//cudaMemset(d_resL2, 0, niter * sizeof(float));

		// texture object
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		cudaMalloc3DArray(&d_x_array, &channelDesc, make_cudaExtent(geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ));
		cudaResourceDesc resDesc = {};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = d_x_array;
		cudaTextureDesc texDesc = {};
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.addressMode[2] = cudaAddressModeClamp;
		texDesc.filterMode = cudaFilterModePoint;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;
		cudaCreateTextureObject(&tex_x, &resDesc, &texDesc, nullptr);

		cudaChannelFormatDesc channelDesc_r_aux_1 = cudaCreateChannelDesc<float>();
		cudaMalloc3DArray(&d_p_array, &channelDesc_r_aux_1, make_cudaExtent(geo.maxnCamV, geo.maxnCamU, 3 * geo.numCam));
		cudaResourceDesc resDesc_p = {};
		resDesc_p.resType = cudaResourceTypeArray;
		resDesc_p.res.array.array = d_p_array;
		cudaTextureDesc texDesc_p = {};
		texDesc_p.addressMode[0] = cudaAddressModeClamp;
		texDesc_p.addressMode[1] = cudaAddressModeClamp;
		texDesc_p.addressMode[2] = cudaAddressModeClamp;
		texDesc_p.filterMode = cudaFilterModeLinear;
		texDesc_p.readMode = cudaReadModeElementType;
		texDesc_p.normalizedCoords = 0;
		cudaCreateTextureObject(&tex_p, &resDesc_p, &texDesc_p, nullptr);

		// for Ax
		cudaMallocHost((void**)&projParamsArrayHost, 4 * geo.numCam * sizeof(Point3D));
		cudaMalloc(&projParamsArrayDev, 4 * geo.numCam * sizeof(Point3D));

		Point3D source, deltaU, deltaV, uvOrigin;
		for (int proj_j = 0; proj_j < geo.numCam; proj_j++) {
			computeDeltas_Siddon(geo, proj_j, &uvOrigin, &deltaU, &deltaV, &source);
			projParamsArrayHost[4 * proj_j + 0] = uvOrigin;
			projParamsArrayHost[4 * proj_j + 1] = deltaU;
			projParamsArrayHost[4 * proj_j + 2] = deltaV;
			projParamsArrayHost[4 * proj_j + 3] = source;
		}

		cudaMemcpy(projParamsArrayDev, projParamsArrayHost,4 * geo.numCam * sizeof(Point3D),cudaMemcpyHostToDevice);

		// for AtbVal_fun
		Point3D xyzOrigin, midPtBK, midDirBK;
		float DSD, co;

		cudaMallocHost((void**)&projParamsArray2Host, 6 * geo.numCam * sizeof(Point3D));
		cudaMalloc((void**)&projParamsArray2Dev, 6 * geo.numCam * sizeof(Point3D));
		cudaMallocHost((void**)&projCoeffArray2Host, 2 * geo.numCam * sizeof(float));
		cudaMalloc((void**)&projCoeffArray2Dev, 2 * geo.numCam * sizeof(float));

		for (unsigned int proj_j = 0; proj_j < geo.numCam; proj_j++) {
			computeDeltasCube(geo, proj_j, &xyzOrigin, &source, &midPtBK, &midDirBK, &deltaU, &deltaV);
			projParamsArray2Host[6 * proj_j] = xyzOrigin;		// 7*j because we have 7 Point3D values per projection
			projParamsArray2Host[6 * proj_j + 1] = source;
			projParamsArray2Host[6 * proj_j + 2] = midPtBK;
			projParamsArray2Host[6 * proj_j + 3] = midDirBK;
			projParamsArray2Host[6 * proj_j + 4] = deltaU;
			projParamsArray2Host[6 * proj_j + 5] = deltaV;

			computeBackgroundCoef(geo, proj_j, source, midPtBK, &DSD, &co);

			projCoeffArray2Host[2 * proj_j] = DSD;
			projCoeffArray2Host[2 * proj_j + 1] = co;
		}
		cudaMemcpy(projParamsArray2Dev, projParamsArray2Host, sizeof(Point3D) * 6 * geo.numCam, cudaMemcpyHostToDevice);
		cudaMemcpy(projCoeffArray2Dev, projCoeffArray2Host, sizeof(float) * 2 * geo.numCam, cudaMemcpyHostToDevice);


		


		// ---------------- streams & events ----------------
		cvStreams.resize(geo.numCam);
		streams.resize(geo.numCam);
		events.resize(geo.numCam);
		for (int k = 0; k < geo.numCam; ++k) {
			streams[k] = static_cast<cudaStream_t>(cvStreams[k].cudaPtr());
			cudaEventCreate(&events[k]);
		}

		cudaStreamCreate(&reconstructionStream);
		for (int i = 0; i < 3; ++i)
			cudaStreamCreate(&stream_diff[i]);

		// ---------------- auxiliary buffers ----------------
		
	
	}

	void freeAll() {
		for (int k = 0; k < geo.numCam; ++k) {
			cudaEventDestroy(events[k]);
			cudaStreamDestroy(streams[k]);
		}
		cudaStreamDestroy(reconstructionStream);
		for (int i = 0; i < 3; ++i)
			cudaStreamDestroy(stream_diff[i]);

		cudaDestroyTextureObject(tex_x);
		cudaDestroyTextureObject(tex_p);
		cudaFree(d_x_array);
		cudaFree(d_p_array);
		cudaFree(d_x);
		cudaFree(d_W);
		cudaFree(d_U);
		cudaFree(d_V);
		cudaFree(d_projectionsob);
		cudaFree(d_IMCam);
		cudaFree(d_RrCam);
		cudaFree(d_Zbc);
		cudaFree(d_Zpc);
		cudaFree(d_KR_inv);
		cudaFree(d_scale);

		cudaFree(d_prox_aux_1);
		cudaFree(d_prox_aux_2);
		cudaFree(d_q_aux_1);
		cudaFree(d_q_aux_2);
		cudaFree(d_r_aux_1);
		cudaFree(d_r_aux_2);
		cudaFree(d_image);
		cudaFree(d_p_aux_1);
		cudaFree(d_p_aux_2);
		cudaFree(d_s_aux_1);
		cudaFree(d_s_aux_2);
		cudaFree(d_p);
		cudaFree(d_s);
		cudaFree(d_gamma);
		cudaFree(d_gamma1);
		cudaFree(d_aux_gamma);
		cudaFree(d_gamma_q_aux_1);
		cudaFree(d_gamma_q_aux_2);
		cudaFree(d_alpha);
		cudaFree(d_beta);
		cudaFree(d_resL2);
		cudaFree(d_condition);

		cudaFreeHost(projParamsArrayHost);
		cudaFree(projParamsArrayDev);
		cudaFree(projParamsArray2Dev);
		cudaFreeHost(projParamsArray2Host);
		cudaFree(projCoeffArray2Dev);
		cudaFreeHost(projCoeffArray2Host);

		free(geoArray);
		free(h_condition);
		
		cudaDeviceReset();
	}

};
