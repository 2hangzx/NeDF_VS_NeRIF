/*-------------------------------------------------------------------------
 *
 * Header CUDA functions for ray-voxel intersection based projection
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

#ifndef PROJECTION_HPP_SIDDON
#define PROJECTION_HPP_SIDDON
int siddon_ray_projection(float*  img, Geometry geo, const int diffselect, float** result, const GpuIds& gpuids, Blur blur);

//double computeMaxLength(Geometry geo, double alpha);
void computeDeltas_Siddon(Geometry geo, int i, Point3D* uvorigin, Point3D* deltaU, Point3D* deltaV, Point3D* source, Blur blur);
void splitImage(unsigned int splits, Geometry geo, Geometry* geoArray);
void freeGeoArray(unsigned int splits, Geometry* geoArray);
//double maxDistanceCubeXY(Geometry geo, double alpha,int i);


#endif
#ifndef PROJECTION_HPP
void checkFreeMemory(const GpuIds& gpuids, size_t *mem_GPU_global);
#endif
