/*-------------------------------------------------------------------------
 *
 * MATLAB MEX gateway for back projection
 *
 * This file gets the data from MATLAB, checks it for errors and then
 * parses it to C and calls the relevant C/CUDA fucntions.
 *
---------------------------------------------------------------------------
Copyright (c) 2024, Beihang University
All rights reserved.
Contact: Weihu22@buaa.edu.cn
Codes  : XXXX
---------------------------------------------------------------------------
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "../../../Common/CUDA/types_BOSLAB.h"
#include "../../../Common/CUDA/GpuIds.hpp"
#include "../../../Common/CUDA/BOSLAB_common.h"
#include "../../../Common/CUDA/voxel_backprojection.h"

 /**
   * MEX gateway
   */

void mexFunction(int  nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	//     clock_t begin, end;
	//     begin = clock();


		//Check amount of inputs
	if (nrhs != 5) {
		mexPrintf("BOSLAB:MEX:Atb:InvalidInput", "Invalid number of inputs to MEX file.");
	}
	////////////////////////////
	// 5th argument is array of GPU-IDs.
	GpuIds gpuids;
	{
		size_t iM = mxGetM(prhs[4]);
		if (iM != 1) {
			mexPrintf("BOSLAB:MEX:Atb:unknown", "5th parameter must be a row vector.");
			return;
		}
		size_t uiGpuCount = mxGetN(prhs[4]);
		if (uiGpuCount == 0) {
			mexPrintf("BOSLAB:MEX:Ax:unknown", "5th parameter must be a row vector.");
			return;
		}
		int* piGpuIds = (int*)mxGetData(prhs[4]);
		gpuids.SetIds(uiGpuCount, piGpuIds);
	}
	////////////////////////////
	// 4th argument is interpolated or ray-voxel/Siddon
	
	if (mxIsChar(prhs[3]) != 1)
		mexPrintf("BOSLAB:MEX:Atb:InvalidInput", "4rd input should be a string");

	/* copy the string data from prhs[0] into a C string input_ buf.    */
	char *ptype = mxArrayToString(prhs[3]);
	
	if (strcmp(ptype, "matched") && strcmp(ptype, "mask")){
		mexPrintf("BOSLAB:MEX:Atb:InvalidInput", "4rd input should be 'matched' ");		
	}
	bool pseudo_mask = false;
	if (!strcmp(ptype, "mask")) {
		pseudo_mask = true;
	}

	///////////////////////// 3rd argument: flow gradient direction in 0 or 1 or 2 or 3.
	mxArray const * diffselectArray = prhs[2];
	const int diffselect = static_cast<int>(mxGetScalar(diffselectArray));
	if (diffselect < 0) {
		mexPrintf("BOSLAB:MEX:Ax:InvalidInput", "3rd input should be either '0' or '1' or '2' or '3'");
	}
	////////////////////////// First input.
	// First input should be x from (Ax=b), or the image.
	mxArray const * const projArray = prhs[0];
	mwSize const numDims = mxGetNumberOfDimensions(projArray);


	// Now that input is ok, parse it to C data types.
	float  *  projections = static_cast<float  *>(mxGetData(projArray));
	// We need a float image, and, unfortunatedly, the only way of casting it is by value
	const mwSize *size_proj = mxGetDimensions(projArray); //get size of image



	///////////////////// Second input argument,
	// Geometry structure that has all the needed geometric data.


	mxArray * geometryMex = (mxArray*)prhs[1];

	// IMPORTANT-> Make sure Matlab creates the struct in this order.
	const char *fieldnames[20];
	fieldnames[0] = "nVoxel";
	fieldnames[1] = "sVoxel";
	fieldnames[2] = "dVoxel";
	fieldnames[3] = "Opr";
	fieldnames[4] = "numCam";
	fieldnames[5] = "nCam";
	fieldnames[6] = "dCam";
	fieldnames[7] = "sCam";
	fieldnames[8] = "fCam";
	fieldnames[9] = "Zbc";
	fieldnames[10] = "Zpc";
	fieldnames[11] = "fd";
	fieldnames[12] = "IMCam";
	fieldnames[13] = "RCam";
	fieldnames[14] = "TCam";
	fieldnames[15] = "RrCam";
	fieldnames[16] = "TrCam";
	fieldnames[17] = "Ocr";
	fieldnames[18] = "PbkCorner";
	fieldnames[19] = "accuracy";
	// Now we know that all the input struct is good! Parse it from mxArrays to
	// C structures that MEX can understand.
	double *nVoxel, *sVoxel, *dVoxel, *Opr; //we need to cast these to int
	double *numCam, *nCam, *dCam, *sCam, *fCam, *Zbc, *Zpc, *fd;
	double *IMCam, *RCam, *TCam, *RrCam, *TrCam, *Ocr, *PbkCorner;
	double *acc;
	mxArray    *tmp;
	Geometry geo;
	geo.unitX = 1; geo.unitY = 1; geo.unitZ = 1;

	for (int ifield = 0; ifield < 20; ifield++) {
		tmp = mxGetField(geometryMex, 0, fieldnames[ifield]);
		if (tmp == NULL) {
			//tofix
			continue;
		}
		switch (ifield) {
		case 0:
			nVoxel = (double *)mxGetData(tmp);
			// copy data to MEX memory
			geo.nVoxelX = (int)nVoxel[0];
			geo.nVoxelY = (int)nVoxel[1];
			geo.nVoxelZ = (int)nVoxel[2];
			break;
		case 1:
			sVoxel = (double *)mxGetData(tmp);
			geo.sVoxelX = (float)sVoxel[0];
			geo.sVoxelY = (float)sVoxel[1];
			geo.sVoxelZ = (float)sVoxel[2];
			break;
		case 2:
			dVoxel = (double *)mxGetData(tmp);
			geo.dVoxelX = (float)dVoxel[0];
			geo.dVoxelY = (float)dVoxel[1];
			geo.dVoxelZ = (float)dVoxel[2];
			break;
		case 3:
			Opr = (double *)mxGetData(tmp);
			geo.OprX = (float)Opr[0];
			geo.OprY = (float)Opr[1];
			geo.OprZ = (float)Opr[2];
			break;
		case 4:
			numCam = (double *)mxGetData(tmp);
			geo.numCam = (int)numCam[0];
			break;
		case 5:
			geo.nCamU = (int*)malloc(geo.numCam * sizeof(int));
			geo.nCamV = (int*)malloc(geo.numCam * sizeof(int));

			nCam = (double *)mxGetData(tmp);

			for (int i = 0; i < geo.numCam; i++) {
				geo.nCamU[i] = (int)nCam[i * 2];
				geo.nCamV[i] = (int)nCam[i * 2 + 1];
			}
			break;
		case 6:
			geo.dCamU = (float*)malloc(geo.numCam * sizeof(float));
			geo.dCamV = (float*)malloc(geo.numCam * sizeof(float));
			dCam = (double *)mxGetData(tmp);
			for (int i = 0; i < geo.numCam; i++) {
				geo.dCamU[i] = (float)dCam[i * 2];
				geo.dCamV[i] = (float)dCam[i * 2 + 1];
			}
			break;
		case 7:
			geo.sCamU = (float*)malloc(geo.numCam * sizeof(float));
			geo.sCamV = (float*)malloc(geo.numCam * sizeof(float));

			sCam = (double *)mxGetData(tmp);

			for (int i = 0; i < geo.numCam; i++) {
				geo.sCamU[i] = (float)sCam[i * 2];
				geo.sCamV[i] = (float)sCam[i * 2 + 1];
			}
			break;

		case 8:
			geo.fCam = (int*)malloc(geo.numCam * sizeof(int));

			fCam = (double *)mxGetData(tmp);
			for (int i = 0; i < geo.numCam; i++) {
				geo.fCam[i] = (int)fCam[i];
			}
			break;
		case 9:
			geo.Zbc = (float*)malloc(geo.numCam * sizeof(float));

			Zbc = (double *)mxGetData(tmp);
			for (int i = 0; i < geo.numCam; i++) {
				geo.Zbc[i] = (float)Zbc[i];
			}
			break;
		case 10:
			geo.Zpc = (float*)malloc(geo.numCam * sizeof(float));

			Zpc = (double *)mxGetData(tmp);
			for (int i = 0; i < geo.numCam; i++) {
				geo.Zpc[i] = (float)Zpc[i];
			}
			break;
		case 11:
			geo.fd = (float*)malloc(geo.numCam * sizeof(float));

			fd = (double *)mxGetData(tmp);
			for (int i = 0; i < geo.numCam; i++) {
				geo.fd[i] = (float)fd[i];
			}
			break;
		case 12:
			geo.IMCam = (float*)malloc(9 * geo.numCam * sizeof(float));
			
			IMCam = (double *)mxGetData(tmp);
			for (int i = 0; i < 9 * geo.numCam; i++) {
				geo.IMCam[i] = (float)IMCam[i];
			}
			break;
		case 13:
			geo.RCam = (float*)malloc(9 * geo.numCam * sizeof(float));
			
			RCam = (double *)mxGetData(tmp);
			for (int i = 0; i < 9 * geo.numCam; i++) {
				geo.RCam[i] = (float)RCam[i];
			}
			break;
		case 14:
			geo.TCam = (float*)malloc(3*geo.numCam * sizeof(float));
			
			TCam = (double *)mxGetData(tmp);
			for (int i = 0; i < 3*geo.numCam; i++) {
				geo.TCam[i] = (float)TCam[i];
			}
			break;
		case 15:
			geo.RrCam = (float*)malloc(9 * geo.numCam * sizeof(float));
			
			RrCam = (double *)mxGetData(tmp);
			for (int i = 0; i < 9 * geo.numCam; i++) {
				geo.RrCam[i] = (float)RrCam[i];
			}
			break;
		case 16:
			geo.TrCam = (float*)malloc(3*geo.numCam * sizeof(float));
			
			TrCam = (double *)mxGetData(tmp);
			for (int i = 0; i < 3*geo.numCam; i++) {
				geo.TrCam[i] = (float)TrCam[i];
			}
			break;

		case 17:
			geo.OcrX = (float*)malloc(geo.numCam * sizeof(float));
			geo.OcrY = (float*)malloc(geo.numCam * sizeof(float));
			geo.OcrZ = (float*)malloc(geo.numCam * sizeof(float));

			Ocr = (double *)mxGetData(tmp);
			for (int i = 0; i < geo.numCam; i++) {
				geo.OcrX[i] = (float)Ocr[i * 3];
				geo.OcrY[i] = (float)Ocr[i * 3 + 1];
				geo.OcrZ[i] = (float)Ocr[i * 3 + 2];
			}
			break;

		case 18:
			geo.PbkCornerX = (float*)malloc(4 * geo.numCam * sizeof(float));
			geo.PbkCornerY = (float*)malloc(4 * geo.numCam * sizeof(float));
			geo.PbkCornerZ = (float*)malloc(4 * geo.numCam * sizeof(float));

			PbkCorner = (double *)mxGetData(tmp);
			for (int i = 0; i < 4 * geo.numCam; i++) {
				geo.PbkCornerX[i] = (float)PbkCorner[i * 3];
				geo.PbkCornerY[i] = (float)PbkCorner[i * 3 + 1];
				geo.PbkCornerZ[i] = (float)PbkCorner[i * 3 + 2];
			}
			break;
		case 19:
			acc = (double*)mxGetData(tmp);
			if (acc[0] < 0.001)
				mexPrintf("BOSLAB:MEX:Ax:Accuracy", "Accuracy should be bigger than 0.001");

			geo.accuracy = (float)acc[0];
			break;
		default:
			mexPrintf("BOSLAB:MEX:Ax:unknown", "This should not happen. Weird");
			break;

		}
	}

	geo.maxnCamU = *std::max_element(geo.nCamU, geo.nCamU + geo.numCam);
	geo.maxnCamV = *std::max_element(geo.nCamV, geo.nCamV + geo.numCam);

	//printData(numDims, size_flow, geo, diffselect, ptype, gpuids);


	mwSize outsize[3];
	outsize[0] = geo.nVoxelX;
	outsize[1] = geo.nVoxelY;
	outsize[2] = geo.nVoxelZ;

	plhs[0] = mxCreateNumericArray(3, outsize, mxSINGLE_CLASS, mxREAL);
	float *result = (float *)mxGetPr(plhs[0]);  // WE will NOT be freeing this pointer!

	voxel_backprojection(projections, geo, diffselect, result, pseudo_mask, gpuids);

	return;

}