#include "mat.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "BOSLAB_common.h"
#include "imgtodisplacement.h"
//#include <E:/github_upload/boslab-v2/Common/CUDA/gpu_buffer_pool.h>

struct GPUBufferPool;
void img_to_reconstruction(
	const Geometry& geo,
	const char* filepath,
	const GpuIds& gpuids,
	float* result,
	const ReconstructionPara& reconP,
	GPUBufferPool& gpuPool);
#pragma once