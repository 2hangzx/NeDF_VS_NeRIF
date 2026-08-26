/*-------------------------------------------------------------------------
 *
 * Header CUDA functions for linear interpolation intersection based projection
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
#include "types_BOSLAB.h"
#include "GpuIds.hpp"


#ifndef PROJECTION_HPP
#define PROJECTION_HPP

int interpolation_projection(float* img, Geometry geo, const int diffselect, bool linearBeam, bool EFtracing, float** result, const GpuIds& gpuids, Blur blur);
void computeDeltas(Geometry geo, unsigned int i, Point3D* uvorigin, Point3D* deltaU, Point3D* deltaV, Point3D* source, Blur blur);
void splitImageInterp(unsigned int splits, Geometry geo, Geometry* geoArray);
void freeGeoArray(unsigned int splits, Geometry* geoArray);
void checkFreeMemory(const GpuIds& gpuids, size_t *mem_GPU_global);
#endif