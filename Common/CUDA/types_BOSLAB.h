/*-------------------------------------------------------------------------
 *
 * Header  functions of BOSLAB for defining data structure 
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

#ifndef TYPES_CBCT
#define TYPES_CBCT
struct  Geometry {
	
	//Parameters part of the flow geometry
	int   nVoxelX, nVoxelY, nVoxelZ;
	float sVoxelX, sVoxelY, sVoxelZ;
	float dVoxelX, dVoxelY, dVoxelZ;
	float OprX, OprY, OprZ;
	//Parameters  of the Camera.
	int numCam;
	int   *nCamU, *nCamV;
	float *dCamU, *dCamV;
	float *sCamU, *sCamV;
	float *OcrX, *OcrY, *OcrZ;
	int* fCam;
	float* Zbc;
	float* Zpc;
	float* fd;
	float *IMCam;
	float *RCam;
	float *TCam;
	float *RrCam;
	float *TrCam;
	int maxnCamU, maxnCamV;
	//Parameters  of the Background.
	float *PbkCornerX, *PbkCornerY, *PbkCornerZ;
	// The base unit we are working with in mm. 
	float unitX;
	float unitY;
	float unitZ;

	//User option
	float accuracy;
};

struct Point3D {
	float x;
	float y;
	float z;
};

struct ReconstructionPara {
	float lambda;
	int niter_outer;
	int niter;
	int niter_break;
};

struct  Blur {
	int flag;
	int num;
	float *DeltaX, *DeltaY, *DeltaZ;
};



#endif
