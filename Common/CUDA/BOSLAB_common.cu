/*-------------------------------------------------------------------------
 *
 * CUDA functions for BOSLAB common
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
#include "BOSLAB_common.h"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>


#define cudaCheckErrors(msg) \
do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
                mexPrintf("%s \n",msg);\
                cudaDeviceReset();\
                mexPrintf("BOSLAB:common",cudaGetErrorString(__err));\
        } \
} while (0)

//______________________________________________________________________________
//
//      Function:       freeGeoArray
//
//      Description:    Frees the memory from the geometry array for multiGPU.
//______________________________________________________________________________
void freeGeoArray(unsigned int splits, Geometry* geoArray) {
	free(geoArray);
}
//______________________________________________________________________________
//
//      Function:       checkFreeMemory
//
//      Description:    check available memory on devices
//______________________________________________________________________________
void checkFreeMemory(const GpuIds& gpuids, size_t *mem_GPU_global) {
	size_t memfree;
	size_t memtotal;
	int deviceCount = gpuids.GetLength();
	for (int dev = 0; dev < deviceCount; dev++) {
		cudaSetDevice(gpuids[dev]);
		cudaMemGetInfo(&memfree, &memtotal);
		if (dev == 0) *mem_GPU_global = memfree;
		if (memfree < memtotal / 2) {
			mexPrintf("tvDenoise:tvdenoising:GPU", "One (or more) of your GPUs is being heavily used by another program (possibly graphics-based).\n Free the GPU to run TIGRE\n");
		}
		cudaCheckErrors("Check mem error");
		*mem_GPU_global = (memfree < *mem_GPU_global) ? memfree : *mem_GPU_global;
	}
	*mem_GPU_global = (size_t)((double)*mem_GPU_global*0.95);

	//*mem_GPU_global= insert your known number here, in bytes.
}


//______________________________________________________________________________
//
//      Function:       printData
//
//      Description:    print input Geo data
//______________________________________________________________________________
void printData(mwSize const numDims, const mwSize *size_flow, const Geometry& geo,
	const int diffselect, char* ptype, GpuIds &gpuids) {
	/////////////////////////////////////// img size
	mexPrintf("Flow/Proj dimensions: (");
	for (mwSize i = 0; i < numDims; ++i) {
		mexPrintf("%d", size_flow[i]);
		if (i < numDims - 1) {
			mexPrintf(", ");
		}
	}
	mexPrintf(")\n");

	/////////////////////////////////////// geo
	mexPrintf("Geo: \n");
	//probe area
	mexPrintf("nVoxel: ( %d, %d, %d )\n", geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ);
	mexPrintf("sVoxel: ( %f, %f, %f )\n", geo.sVoxelX, geo.sVoxelY, geo.sVoxelZ);
	mexPrintf("dVoxel: ( %f, %f, %f )\n", geo.dVoxelX, geo.dVoxelY, geo.dVoxelZ);
	mexPrintf("Opr: ( %f, %f, %f )\n", geo.OprX, geo.OprY, geo.OprZ);

	//Camera
	mexPrintf("numCam: ( %d )\n", geo.numCam);

	//nCam
	for (size_t i = 0; i < geo.numCam; ++i) {
		if (i == 0) mexPrintf("nCamU: [  ");
		mexPrintf("%d ", geo.nCamU[i]);
	}
	mexPrintf("  ]\n");

	for (size_t i = 0; i < geo.numCam; ++i) {
		if (i == 0) mexPrintf("nCamV: [  ");
		mexPrintf("%d ", geo.nCamV[i]);
	}
	mexPrintf("  ]\n");

	//dCam
	for (size_t i = 0; i < geo.numCam; ++i) {
		if (i == 0) mexPrintf("dCamU: [  ");
		mexPrintf("%f ", geo.dCamU[i]);
	}
	mexPrintf("  ]\n");

	for (size_t i = 0; i < geo.numCam; ++i) {
		if (i == 0) mexPrintf("dCamV: [  ");
		mexPrintf("%f ", geo.dCamV[i]);
	}
	mexPrintf("  ]\n");

	//sCam
	for (size_t i = 0; i < geo.numCam; ++i) {
		if (i == 0) mexPrintf("sCamU: [  ");
		mexPrintf("%f ", geo.sCamU[i]);
	}
	mexPrintf("  ]\n");

	for (size_t i = 0; i < geo.numCam; ++i) {
		if (i == 0) mexPrintf("sCamV: [  ");
		mexPrintf("%f ", geo.sCamV[i]);
	}
	mexPrintf("  ]\n");

	//Ocr
	for (size_t i = 0; i < geo.numCam; ++i) {
		if (i == 0) mexPrintf("OcrX: [  ");
		mexPrintf("%f ", geo.OcrX[i]);
	}
	mexPrintf("  ]\n");

	for (size_t i = 0; i < geo.numCam; ++i) {
		if (i == 0)  mexPrintf("OcrY: [  ");
		mexPrintf("%f ", geo.OcrY[i]);
	}
	mexPrintf("  ]\n");

	for (size_t i = 0; i < geo.numCam; ++i) {
		if (i == 0) mexPrintf("OcrZ: [  ");
		mexPrintf("%f ", geo.OcrZ[i]);
	}
	mexPrintf("  ]\n");

	//fCam
	mexPrintf("fCam: [  ");
	for (size_t i = 0; i < geo.numCam; ++i) {
		mexPrintf("%d ", geo.fCam[i]);
	}
	mexPrintf("  ]\n");

	//Zbc
	mexPrintf("Zbc: [  ");
	for (size_t i = 0; i < geo.numCam; ++i) {
		mexPrintf("%f ", geo.Zbc[i]);
	}
	mexPrintf("  ]\n");

	//Zpc
	mexPrintf("Zpc: [  ");
	for (size_t i = 0; i < geo.numCam; ++i) {
		mexPrintf("%f ", geo.Zpc[i]);
	}
	mexPrintf("  ]\n");

	//fd
	mexPrintf("fd: [  ");
	for (size_t i = 0; i < geo.numCam; ++i) {
		mexPrintf("%f ", geo.fd[i]);
	}
	mexPrintf("  ]\n");

	//IMCam
	for (size_t i = 0; i < 9 * geo.numCam; ++i) {
		if (i == 0) mexPrintf("IMCam: [  ");
		mexPrintf("%f ", geo.IMCam[i]);
	}
	mexPrintf("  ]\n");


	//RCam
	for (size_t i = 0; i < 9 * geo.numCam; ++i) {
		if (i == 0) mexPrintf("RCam: [  ");
		mexPrintf("%f ", geo.RCam[i]);
	}
	mexPrintf("  ]\n");

	

	//TCam
	for (size_t i = 0; i < 3*geo.numCam; ++i) {
		if (i == 0) mexPrintf("TCam: [  ");
		mexPrintf("%f ", geo.TCam[i]);
	}
	mexPrintf("  ]\n");


	//TrCam
	for (size_t i = 0; i < 3*geo.numCam; ++i) {
		if (i == 0) mexPrintf("TrCam: [  ");
		mexPrintf("%f ", geo.TrCam[i]);
	}
	mexPrintf("  ]\n");



	//RrCam
	for (size_t i = 0; i < 9 * geo.numCam; ++i) {
		if (i == 0) mexPrintf("RrCam: [  ");
		mexPrintf("%f ", geo.RrCam[i]);
	}
	mexPrintf("  ]\n");

	

	//maxnCamU
	mexPrintf("maxnCamU: ( %d )\n", geo.maxnCamU);

	//maxnCamV
	mexPrintf("maxnCamV: ( %d )\n", geo.maxnCamV);

	//background
	for (size_t i = 0; i < 4 * geo.numCam; ++i) {
		if (i == 0) mexPrintf("PbkCornerX: [  ");
		mexPrintf("%f ", geo.PbkCornerX[i]);
	}
	mexPrintf("  ]\n");

	for (size_t i = 0; i < 4 * geo.numCam; ++i) {
		if (i == 0) mexPrintf("PbkCornerY: [  ");
		mexPrintf("%f ", geo.PbkCornerY[i]);
	}
	mexPrintf("  ]\n");

	for (size_t i = 0; i < 4 * geo.numCam; ++i) {
		if (i == 0) mexPrintf("PbkCornerZ: [  ");
		mexPrintf("%f ", geo.PbkCornerZ[i]);
	}
	mexPrintf("  ]\n");

	// accuracy
	mexPrintf("accuracy: ( %f )\n", geo.accuracy);

	/////////////////////////////////////// diffselect
	mexPrintf("diffselect: ( %d )\n", diffselect);

	/////////////////////////////////////// ptype
	mexPrintf("ptype: ( %s )\n", ptype);

	/////////////////////////////////////// GPUids.devices
	mexPrintf("GpuLength: ( %d )\n", gpuids.GetLength());
}


void printGeoData(const Geometry& geo) {
	/////////////////////////////////////// geo
	mexPrintf("Geo: \n");
	//probe area
	mexPrintf("nVoxel: ( %d, %d, %d )\n", geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ);
	mexPrintf("sVoxel: ( %f, %f, %f )\n", geo.sVoxelX, geo.sVoxelY, geo.sVoxelZ);
	mexPrintf("dVoxel: ( %f, %f, %f )\n", geo.dVoxelX, geo.dVoxelY, geo.dVoxelZ);
	mexPrintf("Opr: ( %f, %f, %f )\n", geo.OprX, geo.OprY, geo.OprZ);

	//Camera
	mexPrintf("numCam: ( %d )\n", geo.numCam);

	//nCam
	for (size_t i = 0; i < geo.numCam; ++i) {
		if (i == 0) mexPrintf("nCamU: [  ");
		mexPrintf("%d ", geo.nCamU[i]);
	}
	mexPrintf("  ]\n");

	for (size_t i = 0; i < geo.numCam; ++i) {
		if (i == 0) mexPrintf("nCamV: [  ");
		mexPrintf("%d ", geo.nCamV[i]);
	}
	mexPrintf("  ]\n");

	//dCam
	for (size_t i = 0; i < geo.numCam; ++i) {
		if (i == 0) mexPrintf("dCamU: [  ");
		mexPrintf("%f ", geo.dCamU[i]);
	}
	mexPrintf("  ]\n");

	for (size_t i = 0; i < geo.numCam; ++i) {
		if (i == 0) mexPrintf("dCamV: [  ");
		mexPrintf("%f ", geo.dCamV[i]);
	}
	mexPrintf("  ]\n");

	//sCam
	for (size_t i = 0; i < geo.numCam; ++i) {
		if (i == 0) mexPrintf("sCamU: [  ");
		mexPrintf("%f ", geo.sCamU[i]);
	}
	mexPrintf("  ]\n");

	for (size_t i = 0; i < geo.numCam; ++i) {
		if (i == 0) mexPrintf("sCamV: [  ");
		mexPrintf("%f ", geo.sCamV[i]);
	}
	mexPrintf("  ]\n");

	//Ocr
	for (size_t i = 0; i < geo.numCam; ++i) {
		if (i == 0) mexPrintf("OcrX: [  ");
		mexPrintf("%f ", geo.OcrX[i]);
	}
	mexPrintf("  ]\n");

	for (size_t i = 0; i < geo.numCam; ++i) {
		if (i == 0)  mexPrintf("OcrY: [  ");
		mexPrintf("%f ", geo.OcrY[i]);
	}
	mexPrintf("  ]\n");

	for (size_t i = 0; i < geo.numCam; ++i) {
		if (i == 0) mexPrintf("OcrZ: [  ");
		mexPrintf("%f ", geo.OcrZ[i]);
	}
	mexPrintf("  ]\n");

	//fCam
	mexPrintf("fCam: [  ");
	for (size_t i = 0; i < geo.numCam; ++i) {
		mexPrintf("%d ", geo.fCam[i]);
	}
	mexPrintf("  ]\n");

	//Zbc
	mexPrintf("Zbc: [  ");
	for (size_t i = 0; i < geo.numCam; ++i) {
		mexPrintf("%f ", geo.Zbc[i]);
	}
	mexPrintf("  ]\n");

	//Zpc
	mexPrintf("Zpc: [  ");
	for (size_t i = 0; i < geo.numCam; ++i) {
		mexPrintf("%f ", geo.Zpc[i]);
	}
	mexPrintf("  ]\n");

	//fd
	mexPrintf("fd: [  ");
	for (size_t i = 0; i < geo.numCam; ++i) {
		mexPrintf("%f ", geo.fd[i]);
	}
	mexPrintf("  ]\n");

	//IMCam
	for (size_t i = 0; i < 9 * geo.numCam; ++i) {
		if (i == 0) mexPrintf("IMCam: [  ");
		mexPrintf("%f ", geo.IMCam[i]);
	}
	mexPrintf("  ]\n");


	//RCam
	for (size_t i = 0; i < 9 * geo.numCam; ++i) {
		if (i == 0) mexPrintf("RCam: [  ");
		mexPrintf("%f ", geo.RCam[i]);
	}
	mexPrintf("  ]\n");


	//TCam
	for (size_t i = 0; i < 3*geo.numCam; ++i) {
		if (i == 0) mexPrintf("TCam: [  ");
		mexPrintf("%f ", geo.TCam[i]);
	}
	mexPrintf("  ]\n");


	//TrCam
	for (size_t i = 0; i < 3*geo.numCam; ++i) {
		if (i == 0) mexPrintf("TrCam: [  ");
		mexPrintf("%f ", geo.TrCam[i]);
	}
	mexPrintf("  ]\n");


	//RrCam
	for (size_t i = 0; i < 9 * geo.numCam; ++i) {
		if (i == 0) mexPrintf("RrCam: [  ");
		mexPrintf("%f ", geo.RrCam[i]);
	}
	mexPrintf("  ]\n");


	//maxnCamU
	mexPrintf("maxnCamU: ( %d )\n", geo.maxnCamU);

	//maxnCamV
	mexPrintf("maxnCamV: ( %d )\n", geo.maxnCamV);

	//background
	for (size_t i = 0; i < 4 * geo.numCam; ++i) {
		if (i == 0) mexPrintf("PbkCornerX: [  ");
		mexPrintf("%f ", geo.PbkCornerX[i]);
	}
	mexPrintf("  ]\n");

	for (size_t i = 0; i < 4 * geo.numCam; ++i) {
		if (i == 0) mexPrintf("PbkCornerY: [  ");
		mexPrintf("%f ", geo.PbkCornerY[i]);
	}
	mexPrintf("  ]\n");

	for (size_t i = 0; i < 4 * geo.numCam; ++i) {
		if (i == 0) mexPrintf("PbkCornerZ: [  ");
		mexPrintf("%f ", geo.PbkCornerZ[i]);
	}
	mexPrintf("  ]\n");

	// accuracy
	mexPrintf("accuracy: ( %f )\n", geo.accuracy);	
}

void printProjectionsData(mwSize const numDims, const mwSize *size_flow) {
	/////////////////////////////////////// img size
	mexPrintf("Flow/Proj dimensions: (");
	for (mwSize i = 0; i < numDims; ++i) {
		mexPrintf("%d", size_flow[i]);
		if (i < numDims - 1) {
			mexPrintf(", ");
		}
	}
	mexPrintf(")\n");
}
