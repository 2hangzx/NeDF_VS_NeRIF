#pragma once

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/cudaarithm.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

struct GPUBufferPool;
void imgtodisplacement(const Geometry& geo,
	const char* filepath,
	const GpuIds& gpuids,
	float* projectionsU,
	float* projectionsV,
	GPUBufferPool& gpuPool);